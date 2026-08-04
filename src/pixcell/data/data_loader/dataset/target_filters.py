
from enum import IntEnum
from pixcell.configs.data_config import AnnotationConfig
 

class ICHClass(IntEnum):
    IVH = 1
    IPH = 2
    SDH = 3
    EDH = 4
    SAH = 5

_TARGET_FILTER = {

    # ///////////////////////Annotations//////////////////////////

    # ======================== ICH ===============================

    #---------------------- Multiclass Seg -----------------------

    'ich_multiclass_segmentation_series' : AnnotationConfig(
        task="ICH",
        builder="multiclass_mask_volume",
        cache_key='all'
    ),

    'ich_multiclass_segmentation_slice' : AnnotationConfig(
        task="ICH",
        builder="multiclass_mask",
        cache_key='all'
    ),   

    'ich_multiclass_segmentation_series_123' : AnnotationConfig(
        task="ICH",
        builder="multiclass_mask_for_classes_volume",
        parameters={
            "classes":[1, 2, 3],
        },
        cache_key='123'
    ),

    'ich_multiclass_segmentation_slice_123' : AnnotationConfig(
        task="ICH",
        builder="multiclass_mask_for_classes",
        parameters={
            "classes":[1, 2, 3],
        },
        cache_key='123'
    ),   


    #------------------------ Binary Seg --------------------------

    # Any ICH
    'any_ich_segmentation_series' : AnnotationConfig(
        task="ICH",
        builder="binary_mask_any_ich_volume",
    ),   

    'any_ich_segmentation_slice' : AnnotationConfig(
        task="ICH",
        builder="binary_mask_any_ich",
    ),   

    # IVH
    'ivh_segmentation_series' : AnnotationConfig(
        task="ICH",
        builder="binary_mask_for_class_volume",
        parameters={
            "class_value":ICHClass.IVH.value,
        },
        cache_key='1'
    ),  

    'ivh_segmentation_slice' : AnnotationConfig(
        task="ICH",
        builder="binary_mask_for_class",
        parameters={
            "class_value":ICHClass.IVH.value,
        },
        cache_key='1'
    ), 

    # IPH
    'iph_segmentation_series' : AnnotationConfig(
        task="ICH",
        builder="binary_mask_for_class_volume",
        parameters={
            "class_value":ICHClass.IPH.value,
        },
        cache_key='2'
    ),  

    'iph_segmentation_slice' : AnnotationConfig(
        task="ICH",
        builder="binary_mask_for_class",
        parameters={
            "class_value":ICHClass.IPH.value,
        },
        cache_key='2'
    ), 

    # SDH
    'sdh_segmentation_series' : AnnotationConfig(
        task="ICH",
        builder="binary_mask_for_class_volume",
        parameters={
            "class_value":ICHClass.SDH.value,
        },
        cache_key='3'
    ),  

    'sdh_segmentation_slice' : AnnotationConfig(
        task="ICH",
        builder="binary_mask_for_class",
        parameters={
            "class_value":ICHClass.SDH.value,
        },
        cache_key='3'
    ), 

    # EDH
    'edh_segmentation_series' : AnnotationConfig(
        task="ICH",
        builder="binary_mask_for_class_volume",
        parameters={
            "class_value":ICHClass.EDH.value,
        },
        cache_key='4'
    ),  

    'edh_segmentation_slice' : AnnotationConfig(
        task="ICH",
        builder="binary_mask_for_class",
        parameters={
            "class_value":ICHClass.EDH.value,
        },
        cache_key='4'
    ), 

    # SAH
    'sah_segmentation_series' : AnnotationConfig(
        task="ICH",
        builder="binary_mask_for_class_volume",
        parameters={
            "class_value":ICHClass.SAH.value,
        },
        cache_key='5'
    ),  

    'sah_segmentation_slice' : AnnotationConfig(
        task="ICH",
        builder="binary_mask_for_class",
        parameters={
            "class_value":ICHClass.SAH.value,
        },
        cache_key='5'
    ), 

    #--------------------- Multi label seg -----------------------

    # Single head, N output channels (one binary mask per ICH class)
    # Loss is computed independently for each channel.

    # config.data.reconstruction.annotation.targets = list(
    #   _SUPERVISED_TASK_FILTER['sah_segmentation_series'],
    #   _SUPERVISED_TASK_FILTER['edh_segmentation_series'],
    #   _SUPERVISED_TASK_FILTER['iph_segmentation_series'],
    #)

    #--------------------- Multi head seg -----------------------

    # N independent segmentation heads (one binary mask per head).
    # Each head predicts a single ICH class using the same target format.

    # config.data.reconstruction.annotation.targets = list(
    #   _SUPERVISED_TASK_FILTER['sah_segmentation_series'],
    #   _SUPERVISED_TASK_FILTER['edh_segmentation_series'],
    #   _SUPERVISED_TASK_FILTER['iph_segmentation_series'],
    #)


    # ======================== MLS ===============================

    # =================== Skull Fracture =========================

    # /////////////////////////Labels/////////////////////////////

    # ======================== ICH ===============================

    #----------- Triage required keys (postprocess) --------------

    "V_EDH": "EDH_area",
    "V_SDH": "SDH_area",
    "V_IPH": "IPH_area",
    "V_SAH": "SAH_area",
    "V_IVH": "IVH_area",

    #--------- Binary classification / Multilabel ----------------

    "Any_ICH": "any_ich",
    "IVH": "IVH",
    "IPH": "IPH",
    "SAH": "SAH", 
    "EDH": "EDH",
    "SDH": "SDH", 
    
    # ======================== MLS ===============================

    #----------------- Triage required keys ----------------------

    "MLS_mm": "midline_shift_mm",

    #----------------- Keypoints Coordinates ---------------------

    "keypoints": "keypoints",

    # =================== Skull Fracture =========================

    #--------------- Triage required keys / ----------------------

    "fracture_prob": "skull_fracture",

    #-------------------- Bbox Coordinates -----------------------

    "bounding_boxes": "bounding_boxes",

    # ======================= Triage =============================

    "triage_class": "triage_class",
   

}
