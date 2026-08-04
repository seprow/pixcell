from sklearn.model_selection import train_test_split, KFold
from pixcell.configs import Config


def patient_level_split(
    patient_ids: list[str],
    config: Config,
) -> tuple[list[str], list[str]]:
    """
    Split patient IDs into train and validation sets.
    based on dicom_series.PatientID (DatasetManifest.patients)
    """

    train_patients, val_patients = train_test_split(
        patient_ids,
        train_size=config.data.train_val_split,
        random_state=config.data.seed,
        shuffle=True,
    )

    return train_patients, val_patients


def patient_level_cv(
    patient_ids: list[str],
    config: Config,
):
    """
    K-Fold split over patient IDs.
    based on dicom_series.PatientID (DatasetManifest.patients)
    """

    kf = KFold(
        n_splits=config.data.cv_n_splits,
        shuffle=True,
        random_state=config.data.seed,
    )

    for train_idx, val_idx in kf.split(patient_ids):

        train_patients = [patient_ids[i] for i in train_idx]
        val_patients = [patient_ids[i] for i in val_idx]

        yield train_patients, val_patients