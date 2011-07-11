modelId = 'color_cs_rect_1';

clear Model fitPar dispPar Result;

[ Model, FitParam, DisplayParam, DataParam ] = loadConfig (modelId);
images = prepare_images (DataParam);
displayImages (images, DataParam);


