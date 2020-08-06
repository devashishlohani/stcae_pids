from seq_exp import *
import time

if __name__ == "__main__":

        start_time = time.time()

        ##----Set path of learned model-----
        # Either Fall Detection or Intrusion Detection

        ##----Fall Detection----
        #learned_model = 'Models/Thermal_Fall/DSTCAE_UpSamp.h5'
        #learned_model = 'Models/Thermal_Fall/DSTCAE_Deconv.h5'
        learned_model = 'Models/Thermal_Fall/DSTCAE_C3D.h5'

        ##----Intrusion Detection----
        # models not released for privacy concerns
        #learned_model = 'Models/Thermal_Intrusion/DSTCAE_UpSamp.h5'
        #learned_model = 'Models/Thermal_Intrusion/DSTCAE_Deconv.h5'
        #learned_model = 'Models/Thermal_Intrusion/DSTCAE_C3D.h5'

        ##--Choose dataset (corresponding to model intrusion or fall)
        #dset = 'Thermal_Intrusion'
        dset = 'Thermal_Fall'

        ##--Choose evaluation measure
        #RE = 'r_sigma'
        RE = 'r_mu'
        #RE = 'r'

        ##--Evaluation type : per_video or all videos
        ## per-video not allowed for Intrusion detection -> because we have videos with only non-intrusion also.
        ## This helps test our model in only non-intrusion classes also. But this raises error more than 1 class is needed to
        ## calculate AUROC/AUPR
        ## Note: should be used in case of animation of a intrusion video

        evaluation_type = 'per_video' # not for intrusion case (except if you want an animation of video)
        #evaluation_type = 'all_videos'

        ## Optional: Animation per video
        do_animate = False
        use_indicative_threshold = False

        ##--Set frame and window size
        img_width, img_height, win_len = 64, 64, 8

        if learned_model == None:
                print('No model path given, please update learned_model variable in dstcae_c3d_main_test.py')
        else:
                dstcae_ae_exp = SeqExp(pre_load=learned_model, dset=dset, win_len=win_len, img_width=img_width, img_height=img_height)
                print(dstcae_ae_exp.model.summary())
                print(dstcae_ae_exp.model.optimizer.get_config())
                dstcae_ae_exp.test(eval_type=evaluation_type, RE_type=RE, animate=do_animate, indicative_threshold_animation=use_indicative_threshold)

        print("Total Time %.2f s or %.2f mins" % (time.time() - start_time, (time.time() - start_time) / 60))