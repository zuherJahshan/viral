import os
from os import path
from os import system
from os import listdir
from os import remove
from pathlib import Path
from Types import *
import numpy as np
from math import ceil
import pickle

import pandas as pd
from typing import List

class FastaSeqGetter(object):
    """
    FastaSeqGetter class This class is aimed to get a list of accessions as presented in the
    https://www.covid19dataportal.org/sequences database, and download them.
    The main function of this class is simply getSeqs

    """
    def __init__(self):
        """
        Initializes all of the parameters needed to contact the database using the shell API.
        """
        self.command: str = "java -jar cdp-file-downloader.jar"
        self.domain: str = "--domain=VIRAL_SEQUENCES"
        self.datatype: str = "--datatype=SEQUENCES"
        self.format: str = "--format=FASTA"
        self.location: str = "--location=../data/tmp/"
        self.email: str = "--email=NONE"
        self.accessions: str = "--accessions="
        self.protocol: str = "--protocol=FTP"
        self.asperaLocation: str = "--asperaLocation=null"

        self.downloaded_files_path: str = "../data/tmp/viral_sequences/sequences/fasta/"
        self.seq_files_location = "../data/raw/"

    def getSeqFilesLoc(self):
        return self.seq_files_location

    def getSeqs(self,
                acc_list: List[Accession]):
        """
        Args:
            A list of accessions as presented in the database.
            ***
            NOTE: This list might be errorfull and as a result not all accessions will be downloaded correctly,
            this is a low-priority assignment, but a mechanism dealing with those errors must be added.
            e.g. accessions as appears in the accessions.tsv file may contain multiple accession IDs this should be
            accounted for. Also, many accession IDs may be incorrect. Given the implementation of that function,
            another existing sequences may be damaged and not downloaded.
            ***
        Returns:
            TODO: The list of all successful accessions downloaded.

        """
        # Save old accessions pattern
        old_accessions = self.accessions

        # Put in a set all fasta files that exists
        fasta_files_list = listdir(self.seq_files_location)
        fasta_files_set = set()
        for fasta_file in fasta_files_list:
            fasta_files_set.add(fasta_file.split(".")[0])

        # Add new accessions, check if already exists, if so please remove them
        accessions_to_download: int = 0
        for i in range(len(acc_list)):
            if acc_list[i] not in fasta_files_set:
                self.accessions += acc_list[i]
                accessions_to_download += 1
                if i < len(acc_list) - 1:
                    self.accessions += ","
            else:
                print("accession {} already exists".format(acc_list[i]))

        if accessions_to_download == 0:
            return

        # execute command
        bash_command = " "
        bash_command = bash_command.join([self.command,
                                          self.domain,
                                          self.datatype,
                                          self.format,
                                          self.location,
                                          self.email,
                                          self.accessions,
                                          self.protocol,
                                          self.asperaLocation
                                          ])
        print(bash_command) # Redundant
        system(bash_command)

        # Retrieve old accessions
        self.accessions = old_accessions

        # Get created file path
        files = listdir(self.downloaded_files_path)

        # Order the sequences in the appropriate files
        self.__orderSeqFiles(self.downloaded_files_path + files[0])

        # Remove the created file
        remove(self.downloaded_files_path + files[0])

    def __newSeqLine(self,
                     line: str):
        return line[0] == ">"

    def __createFastaFile(self,
                          acc_id: str,
                          seq_descriptor: str,
                          seq: str):

        # create path if does not exist
        Path(self.seq_files_location).mkdir(parents=True,
                                            exist_ok=True)

        # create appropriate file
        f = open(self.seq_files_location + acc_id + ".fasta", "w")

        # write necessary information to file
        f.write(seq_descriptor)
        f.write(seq)

        # Close file
        f.close()
        return

    def __getAccId(self,
                   seq_descriptor: str):
        return seq_descriptor.split('|')[1]

    def __orderSeqFiles(self,
                        file_path: str):
        seq_descriptor: str = ""
        seq: str = ""

        # Open file in "read only" mode
        seqs_file = open(file_path, "r")

        # Order files
        first_seq: bool = True
        for line in seqs_file:
            if self.__newSeqLine(line):
                if first_seq:
                    first_seq = False
                else:
                    # Create the appropriate fasta file
                    self.__createFastaFile(self.__getAccId(seq_descriptor),
                                           seq_descriptor,
                                           seq)
                seq_descriptor = line
                seq = ""
            elif first_seq:
                seqs_file.close()
                return
            else:
                seq += line
        if len(seq) > 0:
            self.__createFastaFile(self.__getAccId(seq_descriptor),
                                   seq_descriptor,
                                   seq)

        # Close sequences files
        seqs_file.close()



class DataCollector(object):
    """
    Class for the collection of sequenced data.
    The covid-19 sequenced data is collected from the:
    https://www.covid19dataportal.org/sequences database.

    This class assumes the existance of the file "accessions.tsv" which is a tab separated file which
    contains all metadata regarding the accessions and sequences available to download from the database

    This class provides two functionalities:
    1. getSeqByAcc - which will download the fasta sequence hinted by the accession provided to the path
    data/raw.
    2. getSeqsByLineage - which will download the fasta sequences related to the given lineage (e.g. "B.1.1.7").
    """
    def __init__(self,
                 all_accessions_filename: str = "../accessions.tsv"):
        # Check if the file exists, if not return an exception.
        """
        The class constructor
        1. Initializing the metadata dataframe from the "accessions.tsv" file.
        2. Initializing the set of all already existing sequences that are found in the "data/raw" directory.
        """

        # Random state
        self.rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(133))) # 133 is an arbitrary number, can be changed

        c_all_accessions_filename = all_accessions_filename + ".gz"
        if not path.isfile(all_accessions_filename):
            "The zipped file must exist"
            assert path.isfile(c_all_accessions_filename),\
                "Fatal error, accessions file is missing!"

            # unzip the file
            system("gzip -dk " + c_all_accessions_filename)


        # initialize a dataframe containing the data resides inside all accessions
        dtype={'acc': str,
               'lineage': str,
               'cross_references': str,
               'collection_date': str,
               'country': str,
               'center_name': str,
               'host': str,
               'TAXON': str,
               'coverage': float,
               'phylogeny': str}
        # Create data/raw directory if needed
        raw_data_path = "../data/raw"
        if not os.path.isdir(raw_data_path):
            Path(raw_data_path).mkdir(parents=True,
                                      exist_ok=True)

        self.acc_df = pd.read_csv(all_accessions_filename, delimiter='\t', dtype=dtype)
        self.existing_lineages_filepath = "../data/existing_lineages.pkl"

        # Initializing the sequence getter
        self.seq_getter = FastaSeqGetter()

        # Initialzing the set of all already existing sequences
        self.existing_seqs = set()
        existing_seqs_list = listdir(self.seq_getter.getSeqFilesLoc())
        for seq in existing_seqs_list:
            self.existing_seqs.add(seq.split('.')[0])

    def getAccDF(self) -> pd.DataFrame:
        return self.acc_df

    # Returns if a sequence exists locally
    def existsLocally(self,
               acc_id: str) -> bool:
        """
        simply returns True if the accession already exist in the "data/raw directory". Else returns False.
        """
        return acc_id in self.existing_seqs

    def _downloadSeqsByAcc(self,
                     acc_ids: List[Accession]):

        # Check which of the accessions really need to be downloaded
        not_existing_accs: List[Accession] = []
        for acc_id  in acc_ids:
            if not self.existsLocally(acc_id):
                not_existing_accs.append(acc_id)

        # get sequences from the cloud
        if len(not_existing_accs) > 0:
            self.seq_getter.getSeqs(not_existing_accs)

        # Update the existing sequences set.
        new_accs: Set = set()
        for acc in not_existing_accs:
            if os.path.exists(acc):
                new_accs.add(acc)

        self.existing_seqs.update(new_accs)

    def _flattenDFAccessions(self,
                             df_accessions):
        accessions = []
        for acc_concat in df_accessions:
            accessions += acc_concat.split(" ")
        return accessions

    def getAccessions(self,
                      lineage: Lineage):
        """
        Returns all accessions (that appear in the dataFrame) that have the lineage <lineage>.
        """
        df = self.acc_df[self.acc_df['lineage'] == lineage]
        df = df.dropna(subset=['acc', 'lineage'])
        df_accessions = df['acc'].values   # Each elem in this list is a concatenation of several accessions

        # Flatten accessions
        accessions = self._flattenDFAccessions(df_accessions)

        accessions = self.rs.permutation(accessions)
        return accessions

    def getLineages(self) -> Set:
        return set(self.acc_df['lineage'].unique().tolist())

    def getLocalAccessions(self,
                           lineage: Lineage):
        local_accessions = []
        accessions_set = set(self.getAccessions(lineage))
        for acc in self.existing_seqs:
            if acc in accessions_set:
                local_accessions.append(acc)
        return local_accessions

    def getLineagesAboveThresh(self,
                               thresh: int):
        lins_above_thresh = []
        lineages = self.getLineages()
        for lineage in lineages:
            accs = self.getAccessions(lineage)
            if len(accs) >= thresh:
                lins_above_thresh.append(lineage)
        return lins_above_thresh

    def getLocalAccessionsCount(self,
                                lineage: Lineage):
        return len(self.getLocalAccessions(lineage))

    def downloadSeqsByLineage(self,
                              lineage: str,
                              amount=256):
        """
        If the lineage exists, it downloads all sequences that are related to that lineage that could be found in the
        "accessions.tsv" file.
        Args: 1. A string specifying the lineage

        Returns: None
        """

        accessions = self.getAccessions(lineage)
        amount = min(amount, len(accessions))

        print("{} accessions available for the lineage {}".format(len(accessions), lineage))
        fold: int = 128

        self._updateLocalLineageFile(lineage)
        folds_in_amount: int = ceil(amount / fold)
        for i in range(folds_in_amount):
            if i < folds_in_amount - 1:
                self._downloadSeqsByAcc(accessions[i*fold: (i+1)*fold])
            else:
                self._downloadSeqsByAcc(accessions[i*fold: amount])

    def getLocalLineages(self):
        if os.path.exists(self.existing_lineages_filepath):
            with open(self.existing_lineages_filepath, 'rb') as f:
                return pickle.load(f)
        else:
            return set()

    def _updateLocalLineageFile(self,
                                lineage: str):
        # update existing_lineage file
        existing_lineages = self.getLocalLineages()
        existing_lineages.add(lineage)
        if os.path.exists(self.existing_lineages_filepath):
            os.remove(self.existing_lineages_filepath)
        with open(self.existing_lineages_filepath, "wb") as f:
            pickle.dump(existing_lineages, f)

    def getAccPath(self,
                   acc_id: str):
        """
        Returns the fasta file of the accession if exists, returns empty string if does not exist
        """
        if not self.existsLocally(acc_id):
            return ""
        else:
            return self.seq_getter.getSeqFilesLoc() + acc_id + ".fasta"
