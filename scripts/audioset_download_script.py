from audioset_download import Downloader

def main():
    d = Downloader(
        root_path='../cache/audioset/train', 
        labels=None, 
        n_jobs=2, 
        download_type='unbalanced_train', 
        copy_and_replicate=False
    )

    print('beggining download')
    d.download(format = 'mp3')
    print('finished download')

if __name__ == '__main__':
    main()