%*************************************************************************************************************
% Helper function to crawl through all centroid data files and check the sector, camera and CCD
function check_sector_camera_ccd (dataPath)

    % Get all filenames
    filenames = dir(fullfile(dataPath, './*.csv'));

    cameraList = [];
    ccdList = [];
    for iFile = 1 : length(filenames)
        data = readtable(filenames(iFile).name);
        cameraList  = unique(union(cameraList, unique(data.x_Camera)));
        ccdList     = unique(union(ccdList, unique(data.CCD)));

    end

    display(['Cameras present = ', num2str(cameraList')])
    display(['CCDs present = ', num2str(ccdList')])

end

