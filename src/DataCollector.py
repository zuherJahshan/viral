import os
from os import path
from os import system
from os import listdir
from os import remove
from pathlib import Path
from Types import *
from math import ceil
import pickle
import matplotlib.pyplot as plt

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
            The list of all successful accessions downloaded.

        """
        # Save old accessions pattern
        old_accessions = self.accessions

        # Fill accessions to download
        acc_list_size = len(acc_list)
        cnt = 0
        for acc in acc_list:
            self.accessions += acc
            if cnt < len(acc_list) - 1:
                self.accessions += ","
            cnt += 1

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

        # Get created file path, Order the sequences in the appropriate files and remove the tmp file
        files = listdir(self.downloaded_files_path)
        downloaded_seqs = self.__orderSeqFiles(self.downloaded_files_path + files[0])
        remove(self.downloaded_files_path + files[0])
        return downloaded_seqs

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

        downloaded_accs = []
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
                    downloaded_accs.append(self.__getAccId(seq_descriptor))
                seq_descriptor = line
                seq = ""
            elif first_seq:
                seqs_file.close()
                return []
            else:
                seq += line
        if len(seq) > 0:
            self.__createFastaFile(self.__getAccId(seq_descriptor),
                                   seq_descriptor,
                                   seq)

        # Close sequences files
        seqs_file.close()
        return downloaded_accs

LineageAccessionsMap = Dict[Lineage, Set[Accession]]
AccessionLineageMap = Dict[Accession, Lineage]

class DataCollectorv2(object):
    """
    The data collector is a class that is in charge of downloading sequences from the web to the local computer.
    Also, returns statistics and answers queries about the data, both locally and remotely.
    """
    def __init__(self):
        """
        Builds a dictionary of [lineage -> accessions set]. This dictionary represents all available accessions that
        are found remotely.

        Also, builds/restores Another [lineage -> accessions set] dictionary. The later dict represents all available
        accessions that are found locally.
        """
        self.data_path = "../data/"
        self.raw_data_path = "../data/raw/"

        print("Building Data frame")
        self._configureAccessionsFile()
        self._buildDF()
        print("Done building Data frame")

        # Initializing the sequence getter
        self.seq_getter = FastaSeqGetter()

        # Builds the [lin -> accs], [acc -> lin] remote data dicts.
        print("Building remote dicts")
        self._buildRemoteDicts()
        print("Done building remote dicts")

        # Builds local lin -> accs dict
        print("Building local dicts")
        self._buildLocalDicts()
        print("Done building local dicts")


    def getRemoteLineages(self,
                          accs_thresh: int):
        lins = []
        for lin in self.remote_lin_accs_dict:
            if len(self.remote_lin_accs_dict[lin]) >= accs_thresh:
                lins.append(lin)
        return lins

    def downloadLineages(self,
                         accs_thresh: int,
                         max_accs: int,
                         lineages: List[Lineage] = None) -> None:
        """
        Will download to the computer a maximum of "max_accs" accessions
        of all lineages that has more than "thresh" accessions.

        In case a lineage is already found locally then the functionality is as follows:
            1. if he already has max_accs or more locally, then will be ignored.
            2. if he has less than max_accs locally, then accessions will be downloaded till reaches a maximum of max_accs.
        """
        # Iterate over all remote lineages.
        iterable = None
        if lineages is None:
            iterable = self.remote_lin_accs_dict
        else:
            # Check validity
            for lin in lineages:
                if not lin in self.remote_lin_accs_dict:
                    print("Error: Can't downoload lineage {}. It does not exist in the remote dataset.")
            iterable = lineages

        for lin in iterable:
            # If has less accessions than the threshold, just ignore it
            accs = self.remote_lin_accs_dict[lin]
            if len(accs) < accs_thresh:
                continue
            # If already downloaded and has more than the max accs, also ignore it.
            elif lin in self.local_lin_accs_dict:
                if len(self.local_lin_accs_dict[lin]) >= max_accs:
                    continue
            # Get accessions to download
            accs_to_download = list(accs - self.getLocalAccessions(lin))
            num_of_accs_to_download = min(max_accs - len(self.getLocalAccessions(lin)),
                                          len(accs_to_download) - len(self.getLocalAccessions(lin)))

            # The download will occur in folds of 128 each time
            fold = 128
            folds_to_download = ceil(num_of_accs_to_download / fold)
            for i in range(folds_to_download):
                if i < folds_to_download - 1:
                    self.seq_getter.getSeqs(accs_to_download[i*fold: (i+1)*fold])
                else:
                    self.seq_getter.getSeqs(accs_to_download[i*fold: num_of_accs_to_download])
        self._buildLocalDicts()

    def getLocalAccessions(self, lineage) -> Set[Accession]:
        if lineage in self.local_lin_accs_dict:
            return self.local_lin_accs_dict[lineage]
        return set()

    def getLocalLineages(self,
                         accs_thresh: int) -> List[Lineage]:
        lins = []
        for lin in self.local_lin_accs_dict:
            if len(self.local_lin_accs_dict[lin]) >= accs_thresh:
                lins.append(lin)
        return lins

    def getAccPath(self,
                   accession: Accession) -> str:
        """
        Returns the file path of the accession, if accession does not exist return an empty string
        """

        # Check if exists
        if accession in self.local_acc_lin_dict:
            return self.raw_data_path + accession + ".fasta"
        else:
            return ""

    def getCountHist(self):
        # Create a counts sequence for each lineage
        count_seq = []
        for lin in self.remote_lin_accs_dict:
            accs_cnt = min(len(self.remote_lin_accs_dict[lin]), 1024)
            count_seq.append(accs_cnt)

        pd_series = pd.Series(count_seq)
        pd_series.plot.hist(grid=True,
                            bins=16,
                            rwidth=0.9,
                            color='#607c8e')
        plt.title("Lineage accessions count")
        plt.xlabel('Accessions count')
        plt.ylabel('Lineages count')
        plt.grid(axis='y',
                 alpha=0.75)
        plt.savefig("hist.png")

    #######################################
    #######Private member functions########
    #######################################
    def _saveRemoteDicts(self):
        remote_lin_accs_filepath = self.data_path + "remote_lin_accs.pkl"
        remote_acc_lin_filepath = self.data_path + "remote_acc_lin.pkl"
        if os.path.exists(remote_lin_accs_filepath):
            return
        with open(remote_lin_accs_filepath, 'wb') as f:
            pickle.dump(self.remote_lin_accs_dict,
                        f,
                        pickle.HIGHEST_PROTOCOL)
        with open(remote_acc_lin_filepath, 'wb') as f:
            pickle.dump(self.remote_acc_lin_dict,
                        f,
                        pickle.HIGHEST_PROTOCOL)

    # Returns True if successfull, False if files does not exist
    def _loadRemoteDicts(self) -> bool:
        remote_lin_accs_filepath = self.data_path + "remote_lin_accs.pkl"
        remote_acc_lin_filepath = self.data_path + "remote_acc_lin.pkl"
        if os.path.exists(remote_lin_accs_filepath):
            with open(remote_lin_accs_filepath, 'rb') as f:
                self.remote_lin_accs_dict = pickle.load(f)
            with open(remote_acc_lin_filepath, 'rb') as f:
                self.remote_acc_lin_dict = pickle.load(f)
            return True
        return False

    def _buildLocalDicts(self):
        # Initialize an empty dict
        self.local_lin_accs_dict: LineageAccessionsMap = {}
        self.local_acc_lin_dict: AccessionLineageMap = {}

        # List accessions in data directory
        accessions = os.listdir(self.raw_data_path)
        for acc_file in accessions:
            acc = acc_file.split(".")[0]
            lin = self.remote_acc_lin_dict[acc]
            if lin in self.local_lin_accs_dict:     # if exists, just add to the already existing accessions set
                self.local_lin_accs_dict[lin].add(acc)
            else:   # The lin does not exist in the dict
                self.local_lin_accs_dict.update({lin: set([acc])})
            self.local_acc_lin_dict.update({acc: lin})

    def _buildRemoteDicts(self):
        # Check if they exist locally, if yes just load them...
        self.remote_lin_accs_dict: LineageAccessionsMap = {}
        self.remote_acc_lin_dict: AccessionLineageMap = {}
        if self._loadRemoteDicts():
            return
        for index, row in self.df.iterrows():
            lin = row['lineage']
            accs = self._getAccsFromDFRow(row['acc'])
            if lin in self.remote_lin_accs_dict:    # if exists, just add to the already existing accessions set
                for acc in accs:
                    self.remote_lin_accs_dict[lin].add(acc)
            else:   # The lin does not exist in the dict
                self.remote_lin_accs_dict.update({lin: set(accs)})
            for acc in accs:
                self.remote_acc_lin_dict.update({acc: lin})
        self._saveRemoteDicts()


    def _getAccsFromDFRow(self,
                          accs: str):
        return accs.split(" ")

    def _buildDF(self):
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
        if not os.path.isdir(self.raw_data_path):
            Path(self.raw_data_path).mkdir(parents=True,
                                           exist_ok=True)

        self.df = pd.read_csv(self.accessions_file,
                              delimiter='\t',
                              dtype=dtype)
        self.df.dropna(subset=["lineage"], inplace=True)

    def _configureAccessionsFile(self):
        self.accessions_file = "../accessions.tsv"
        c_all_accessions_filename = self.accessions_file + ".gz"
        if not path.isfile(self.accessions_file):
            "The zipped file must exist"
            assert path.isfile(c_all_accessions_filename),\
                "Fatal error, accessions file is missing!"

            # unzip the file
            system("gzip -dk " + c_all_accessions_filename)
