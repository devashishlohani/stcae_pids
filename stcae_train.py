import time
from models import *
from seq_exp import SeqExp

if __name__ == "__main__":
        '''
        These are the training setting. 
        '''
        start_time = time.time()

        ##--Choose dataset (corresponding to model intrusion or fall)
        #dset = 'Thermal_Fall'
        #dset = 'Thermal_Intrusion'

        img_width, img_height, win_len, epochs = 64, 64, 8, 500

        # Model to train
        model, model_name, model_type = DSTCAE_UpSampling(img_width, img_height, win_len)
        #model, model_name, model_type = DSTCAE_Deconv(img_width, img_height, win_len)
        #model, model_name, model_type = DSTCAE_C3D(img_width, img_height, win_len)

        print('model loaded')
        print(model.summary())
        print(model.optimizer.get_config())

        exp_3D = SeqExp(model = model, model_name = model_name, epochs = epochs, win_len = win_len, dset = dset,
                        img_width = img_width, img_height = img_height, batch_size=32)

        exp_3D.set_train_data()
        print(exp_3D.train_data.shape)
        print('data loaded')

        exp_3D.train()

        print("Total Time %.2f s or %.2f mins" % (time.time() - start_time, (time.time() - start_time) / 60))