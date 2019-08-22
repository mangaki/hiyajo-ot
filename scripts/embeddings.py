from .item_encoder import ItemEncoder

class Embedding:
    def __init__(self, npy_filename, posters_paths=None):
        self.npy_filename = npy_filename
        self.posters_paths = posters_paths or []
        
        self.npy = None
        self.work_ids = []
        self.read_work_ids_from_posters_paths()
        
    def open_npy(self):
        self.npy = np.load(self.npy_filename)
    
    def read_posters_paths_from_filename(self, posters_filename):
        self.posters_paths = []
        with open(posters_filename, 'r') as f:
            for line in f:
                filename = os.path.basename(line.strip())
                self.posters_paths.append(os.path.join(POSTERS_DIRECTORY, filename))
        
        self.read_work_ids_from_posters_paths()
    
    def read_work_ids_from_posters_paths(self):
        for p in self.posters_paths:
            work_id = os.path.basename(p)[:-len('.jpg')]
            try:
                self.work_ids.append(int(work_id))
            except ValueError:
                print('[!] Invalid poster data!')
    
    @property
    def identifier(self):
        return os.path.basename(self.npy_filename)[len('embeddings-'):-len('.npy')]
    
    def generate_collage(self, work_directory=None):
        PER_ROW = 4
        if work_directory is None:
            work_directory = POSTERS_DIRECTORY
        target_path = os.path.join(work_directory, 'montage-{}.png'.format(self.identifier))
        if os.path.exists(target_path):
            print('[!] Overwriting a montage')
            os.remove(target_path)
        os.system('montage -density 300 -tile {}x0 -geometry +5+50 -border 10 {} {}'
                 .format(PER_ROW, ' '.join(self.posters_paths), target_path))
    
    def show_collage(self, work_directory=None):
        if work_directory is None:
            work_directory = POSTERS_DIRECTORY
        
        return Image(os.path.join(work_directory, 'montage-{}.png'.format(self.identifier)))
    
    def show_image(self, work_id, work_directory=None):
        if work_directory is None:
            work_directory = POSTERS_DIRECTORY
        
        return Image(os.path.join(work_directory, '{}.jpg'.format(work_id)))
    
    def show_images_grid(self):
        """
            Relatively slow.
        """
        files = []
        try:
            files = []
            for poster_path in self.posters_paths:
                files.append(open(poster_path, 'rb'))
            
            PER_ROW = 3
            # FIXME: can we go smaller than 100 × 500 while preserving a decent visual overview?
            _, axarr = plt.subplots(math.ceil(len(files)/PER_ROW), PER_ROW, figsize=(100, 500))
            for index, file in enumerate(files):
                a = plt.imread(file)
                col = index % PER_ROW
                row = index // PER_ROW
                # FIXME: can we improve the speed?
                axarr[row, col].imshow(a, aspect='equal', interpolation='none')
        finally:
            for file in files:
                file.close()
        
        
    
    def __repr__(self):
        if self.npy is not None:
            return '<Embedding (opened): {} × {}>'.format(*self.npy.shape)
        else:
            return '<Embedding (closed)>'
    
    @classmethod
    def from_filename(cls, embedding_filename, ignore_posters=False):
        emb_basename = os.path.basename(embedding_filename)
        poster_basename = 'paths-' + emb_basename[len('embeddings-'):-len('.npy')] + '.txt'
        posters_filenames = os.path.join(os.path.dirname(embedding_filename), poster_basename)
        
        if not os.access(posters_filenames, os.R_OK) and not ignore_posters:
            raise ValueError('Posters paths are not available!')
        
        embedding = Embedding(embedding_filename)
        if not ignore_posters:
            embedding.read_posters_paths_from_filename(posters_filenames)
        
        return embedding

def generate_mapping(work_directory=None):
    if work_directory is None:
        work_directory = EMBEDDINGS_DIRECTORY
    
    mapping = []
    work_ids_images = {}
    files_list = os.listdir(work_directory)
    
    for filename in files_list:
        if filename.startswith('embeddings-'):
            try:
                embedding = Embedding.from_filename(os.path.join(work_directory, filename))
                for work_id in embedding.work_ids:
                    work_ids_images[work_id] = partial(embedding.show_image, work_id)
                mapping.append(embedding)
                print('[+] {} has been loaded in the database'.format(filename))
            except Exception as e:
                print('[-] {} is a lonely embedding ({})'.format(filename, e))
    
    return mapping, work_ids_images

def merge_and_order_embeddings(embeddings):
    flattened = []
    
    max_work_id = embeddings[0].work_ids[0]
    for embedding in embeddings:
        for index, work_id in enumerate(embedding.work_ids):
            flattened.append((embedding.npy[index].reshape((1, 512)), work_id))
            max_work_id = max(max_work_id, work_id)
    
    ordered_embeddings = sorted(flattened,
                               key=lambda item: item[1])

    nb_works = max_work_id + 1
    C = np.concatenate([x for x, _ in ordered_embeddings])
    work_id_encoder = {y: x for x, (_, y) in enumerate(ordered_embeddings)}
    encoder = ItemEncoder(work_id_encoder)

    return C.astype(np.int16), encoder
