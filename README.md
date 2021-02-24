# elsa
Elsa: Energy based learning for semi-supervised anomaly detection

How to Run ELSApp

    python3 ELSApp_earlystop.py --save_dir ELSApp_final/esnew_0_1 --n_cluster 500 --load_path /home/data/aya/elsa/pretrains_lars_csi/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_0_ratio_pollution_0.0/last.model --optimizer adam --lr 1e-4 --n_epochs 50

How to Run ELSA

    python3 ELSA_earlystop.py --save_dir ELSA_final/esnew_0_1 --n_cluster 100 --load_path /home/data/aya/elsa/pretrains_lars_simclr_csi/cifar10_resnet18_unsup_simclr_one_class_0_ratio_pollution_0.0/last.model --optimizer adam --lr 1e-4 --n_epochs 50
