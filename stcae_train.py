import time
from models import *
from seq_exp import SeqExp

if __name__ == "__main__":
    '''
        These are the training setting. 
    '''
    start_time = time.time()

    #----Choose dataset (corresponding to model intrusion or fall)
    dset = 'Thermal_Dummy'# Dummy fall dataset to demonstrate
    #dset = 'Thermal_Fall'
    #dset = 'Thermal_Intrusion'

    img_width, img_height, win_len, epochs = 64, 64, 8, 2

    #----Model to train-------
    model, model_name, model_type = DSTCAE_UpSampling(img_width, img_height, win_len)
    #model, model_name, model_type = DSTCAE_Deconv(img_width, img_height, win_len)
    #model, model_name, model_type = DSTCAE_C3D(img_width, img_height, win_len)

    print('model architecture loaded')
    print(model.summary())
    print(model.optimizer.get_config())

    dstcae_exp = SeqExp(model=model, model_name=model_name, epochs=epochs, win_len=win_len, dset=dset,
                       img_width=img_width, img_height=img_height, batch_size=32)

    dstcae_exp.set_train_data()
    print(dstcae_exp.train_data.shape)
    print('data loaded')

    dstcae_exp.train()

    print("Total Time %.2f s or %.2f mins" % (time.time() - start_time, (time.time() - start_time) / 60))
