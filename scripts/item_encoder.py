class ItemEncoder:
    def __init__(self, encoder):
        self.decoder = {v: k for k, v in encoder.items()}
        self.encoder = encoder

    def encode(self, item):
        return self.encoder[item]

    def decode(self, item):
        return self.decoder[item]

    def merge(self, other_encoder, keyspace=None):
        if keyspace is None:
            keyspace = other_encoder.encoder.keys()

        # returns the encoder for other_encoder.encode(self.encode(x))
        return ItemEncoder(
            {
                k: other_encoder.encode(v)
                for k, v in self.encoder.items()
                if v in keyspace
            }
        )
