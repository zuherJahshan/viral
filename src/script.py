from CoViT import *
covit = CoViT("proj1")
covit.buildNNModel()
covit.buildDataset(epochs=40,
                   batch_size=64)
covit.train(shuffle_buffer_size=1000,
            size=526*192)
covit.save()
