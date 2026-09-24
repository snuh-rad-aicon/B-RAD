# BUS-CoT held-out evaluation data

BUS-CoT was used as an independent held-out external evaluation set. It was not merged into the B-RAD training, validation, or original OOD test splits.

The B-RAD authors received a collaborator-prepared evaluation package directly from the BUS-CoT team. That package contains the cropped evaluation images, annotations used for quality-controlled evaluation, and query-to-reference retrieval information. It is not redistributed with this repository and is not fetched by `datasets/download.py`.

The final quality-controlled evaluation cohort contains 3,232 queries. Any included ten-query metadata file is only a path/schema validation subset and must not be used to reproduce the reported full-cohort metrics.

To reproduce the BUS-CoT evaluation, please contact the corresponding authors of the [BUS-CoT dataset paper](https://www.nature.com/articles/s41597-026-06702-9) and request the processed B-RAD evaluation package. After receiving it, place its contents under a relative project path such as:

```text
data/raw/buscot/
```

Do not substitute the original raw images for the cropped evaluation images without repeating the preprocessing and annotation checks: the raw image dimensions can differ from the released masks. The B-RAD evaluation used cropped images, tight bounding boxes derived from the masks, top-100 retrieval candidates, and a separately quality-controlled evaluation list.

Fold-level prediction CSV files are private validation artifacts and are not part of the release.
