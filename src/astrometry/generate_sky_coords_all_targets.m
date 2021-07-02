%*************************************************************************************************************
% This function will generate the sky coordinates (R.A. and Decl.) for all targets csv files in a directory.
%
% Inputs:
%   csv_data_path -- [str] Path the the CSV files generated form the Pythong astrometry code
%   pa_data_path -- [str] top level path to the PA tasks files
%
% Outputs:
%   a bunch of CSV files of the sly coordinates, one corresponding to each input csv file
%
%*************************************************************************************************************
function generate_sky_coords_all_targets (csv_data_path, out_path, pa_data_path)

    % Right now this only workd with Sector 10, Camera 1, CCDs 1-4
    paWrapperArray = cell(1,4);

    % Work through each csv file. Check if the PA tasks are already loaded. If not loaded then process the PA data
    filenames = dir(fullfile(csv_data_path, './*.csv'));

    nFiles = length(filenames);
    for iFile = 1 : nFiles
        file = filenames(iFile).name;

        disp(['Working on file ', num2str(iFile), ' of ', num2str(nFiles)])

        % First get data out of the header
        fullFilename = fullfile(csv_data_path, file);
        A = textread(fullFilename, '%s');
        % Target ID
        % Remove the comma
        targetID = A{4}(1:end-1);
        % Sector
        sector = str2num(A{7});

        % Now get the table data
        data = readtable(fullFilename);

        if (sector ~= 10)
            error('This function only works for Sector 10!')
        end

        % Find which CCDs are needed
        cameraList  = unique(data.x_Camera);
        ccdList     = unique(data.CCD);

        % Check that the appropriate PA Wrappers have been instantiated
        % Targets can cross CCD boundaries
        if cameraList ~= 1
            error('This function only works with Camera 1')
        end
        for iCcd = 1 : length(ccdList)
            ccd = ccdList(iCcd);
            if (isempty(paWrapperArray{1,ccd}))
                % Instantiate the PA task data
                paTopTaskPath = fullfile(pa_data_path, ['s10_1.', num2str(ccd)]);
                subTaskName = dir(fullfile(paTopTaskPath, 'st-*'));
                fullPaTaskPath = fullfile(paTopTaskPath, subTaskName.name);
                paWrapperArray{1,ccd} = paWrapperClass(fullPaTaskPath);
            end
        end

        % Compute sky coordinates for this target for each CCD
        ra  = nan(length(data.CCD),1);
        dec = nan(length(data.CCD),1);
        for iCcd = 1 : length(ccdList)
            ccd = ccdList(iCcd);
            thisCcdHere = data.CCD == ccd;
            if (any(thisCcdHere))
                [ra(thisCcdHere), dec(thisCcdHere)] = paWrapperArray{1,ccd}.pix_2_ra_dec_relative(cameraList, ccd, data.row_pixels_(thisCcdHere), ...
                                            data.column_pixels_(thisCcdHere), data.instrumentTime_TJD_(thisCcdHere));
            end
        end

        % Write out the sky coordinates
        % Can use any of the CCD paWrapper objects to write out data
        outFilename = [file(1:end-4), '_ra_dec.csv'];
        paWrapperArray{1,ccd}.write_ra_dec_to_csv(fullfile(out_path, outFilename), ra, dec, data.instrumentTime_TJD_);
        
    end
end
