import model as model                       # Change here if you have a different `model.py` file
import encoder as encoder                   # Change here if you have a different `encoder.py` file
import collate as collate

config = {
    "chkp_name"         : "esmc_small",
    "chkp_path"         : "models",
    "model"             : model.plm_cat_mean_inf,
    "model_args"        : {
        "hla_dim"       : 960,
        "epi_dim"       : 960,
        "head_div"      : 64,
    },
    "encoder"           : encoder.plm_plm_mask_msa_pair_inf,
    "encoder_args"      : {
        "hla_emb_dir"   : "data/emb_hla_esmc_small_0601_fp16",
    },
    "collate_fn"        : collate.pad_and_mask_collate_fn_inf,
    "Data": {
        "hla_path"      : "data/mhc_mapping.csv",
        "hla_args"      : {
            "hla_header": 'HLA_Name',
            "seq_header": 'HLA_Seq',
            "separator" : ",",
        },
        "test_path"     : "data/dataset_demo.csv",
        "test_args"     : {
            "epi_header": 'Epitope',
            "hla_header": 'MHC',
            "tgt_header": 'Target',
            "separator" : ",",
        },
        "num_workers"   : 8,
    },
    "Test": {
        "batch_size"    : 512,
        "chkp_path"     : "models/prepi_esmc_small_e5_s128_f4_fp16.pth",
        "esm_chkp_path" : "models/esmc_300m_2024_12_v0_fp16.pth",
        "plot"          : True,
        "use_compile"   : False,
    },
}
