% This class is a wrapper for the PA module in the SPOC pipeline. It will load in all data from a PA task and set up an object which can be used to perform
% PA-like operations on data.
%
% It can be used to perform raDec2Pix operations on random data. 


classdef paWrapperClass < handle

    properties
        raDec2PixObject = [];
        deltaQuaternions = [];
    end

    %****************************************************
    %****************************************************
    methods

        %*************************************************************************************************************
        % Constructor
        %
        % Inputs:
        % paPath    -- [str] path to the PA task to load data from
        %
        %*************************************************************************************************************

        function obj = paWrapperClass(paPath)
        
            %***************************************************
            % Extract the raDec2Pix object from a PA data object
            % PA expects to run in the PA task directory. So cd to that directory (and accept modified files)
            initialDir = pwd;
            cd(paPath);
        
            % Load in the PA inputsStruct
            inputsStruct = load('pa-inputs-0.mat');
            inputsStruct = inputsStruct.inputsStruct;

            % Set path to spice kernels to current path
            inputsStruct.raDec2PixModel.spiceKernelStruct.spiceKernelDir = paPath;
            
            % This will trace out code from PA to set up relevent classes.
            % Update PA data structure, validate PA inputs and instantiate PA data
            % object. Also initialize the PA state file in the given sub-task
            % directory. Clear the PA data structure once the object has been
            % instantiated.
            tic;
            display('Instantiating paDataClass object...');
            [inputsStruct] = paDataClass.update_inputs(inputsStruct);
            [paDataObject] = paDataClass(inputsStruct);
            clear inputsStruct
            duration = toc;
            
            display(['paDataClass instantiation duration: ' num2str(duration) ...
                ' seconds = '  num2str(duration/60) ' minutes']);

            % paDataClass.raDec2PixObject is now set up.
            obj.raDec2PixObject = paDataObject.raDec2PixObject;
            
            cd(initialDir)

            %***
          % % This code is from @paDataClass.photometric_analysis
          % % Initialize the PA output structure.
          % paDataObject.initialize_pa_output_structure();

          % % Manage PA processing states.
          % paDataObject.manage_pa_processing_states();

            %***************************************************
            % Construct the raDec2Pix object
          % obj.raDec2PixObject = raDec2PixClass(inputsStruct.raDec2PixModel);


            %***************************************************
            % Extract the delta quaternions
            tic;
            obj.extract_delta_quaternions(paDataObject)
            duration = toc;
            
            display(['delta quaternion generation duration: ' num2str(duration) ...
                ' seconds = '  num2str(duration/60) ' minutes']);

        end

        %*************************************************************************************************************
        % Returns the RA and Decl. given the row and column pixels on the specific camera and ccd. 
        % The stored spacecraft attitude and delta quaternions are used.
        %
        %
        % Inputs:
        %   camera      -- [int] Camera number {1:4}
        %   ccd         -- [int] CCD number {1:4}
        %   row         -- [float array(nDatums)] list of row coordinates to compute sky coordinates for
        %   col         -- [float array(nDatums)] list of col coordinates to compute sky coordinates for
        %   timestamps  -- [float array(nDatums)] timestamps in TJD for row and col arrays above
        %
        % Outputs:
        %   ra      -- [float array(nDatums)] Returned Right Acsension
        %   dec     -- [float array(nDatums)] Returned Declination
        % 
        %*************************************************************************************************************

        function [ra, dec] = pix_2_ra_dec_relative(obj, camera, ccd, row, col, timestamps)

            nDatums = length(timestamps);
            ra = nan(nDatums,1);
            dec = nan(nDatums,1);
           %meanQuaternions = repmat([], [nDatums,1]);

            % Extract delta quaternions for the desired timestamps
            % Use nearest neighbor if timestamps do not align with quaternion timestamps
            quaternionTimestamps = obj.deltaQuaternions(camera).timestamps;
            for iTime = 1:nDatums
                [~,idx] = min(abs(quaternionTimestamps - timestamps(iTime)));
                meanQuaternions(iTime) = quaternion([   mean(obj.deltaQuaternions(camera).objects(idx).qr), ...
                                                mean(obj.deltaQuaternions(camera).objects(idx).qi), ...
                                                mean(obj.deltaQuaternions(camera).objects(idx).qj), ...
                                                mean(obj.deltaQuaternions(camera).objects(idx).qk)]);

            end

            [ra, dec] = obj.raDec2PixObject.pix_2_ra_dec_relative(camera, ccd, row, col, timestamps, meanQuaternions);


        end

        %*************************************************************************************************************
        % Returns the row and column pixels for the RA and Decl. given the on the specific camera and ccd. 
        % The stored spacecraft attitude and delta quaternions are used.
        %
        %
        % Inputs:
        %   camera      -- [int] Camera number {1:4}
        %   ccd         -- [int] CCD number {1:4}
        %   ra      -- [float array(nDatums)] Returned Right Acsension
        %   dec     -- [float array(nDatums)] Returned Declination
        %   timestamps  -- [float array(nDatums)] timestamps in TJD for row and col arrays above
        %
        % Outputs:
        %   row         -- [float array(nDatums)] list of row coordinates to compute sky coordinates for
        %   col         -- [float array(nDatums)] list of col coordinates to compute sky coordinates for
        % 
        %*************************************************************************************************************

        function [row, col] = ra_dec_2_pix_relative(obj, camera, ccd, ra, dec, timestamps)

            nDatums = length(timestamps);
            row = nan(nDatums,1);
            col = nan(nDatums,1);
           %meanQuaternions = repmat([], [nDatums,1]);

            % Extract delta quaternions for the desired timestamps
            % Use nearest neighbor if timestamps do not align with quaternion timestamps
            quaternionTimestamps = obj.deltaQuaternions(camera).timestamps;
            for iTime = 1:nDatums
                [~,idx] = min(abs(quaternionTimestamps - timestamps(iTime)));
                meanQuaternions(iTime) = quaternion([   mean(obj.deltaQuaternions(camera).objects(idx).qr), ...
                                                mean(obj.deltaQuaternions(camera).objects(idx).qi), ...
                                                mean(obj.deltaQuaternions(camera).objects(idx).qj), ...
                                                mean(obj.deltaQuaternions(camera).objects(idx).qk)]);

            end

            % ra_dec_2_pix_relative returns a matrix is ra,dec and timestamsp are arrays!
            for idx = 1:nDatums
                [camera(idx), ccd(idx), row(idx), col(idx)] = obj.raDec2PixObject.ra_dec_2_pix_relative(ra(idx), dec(idx), timestamps(idx), meanQuaternions(idx));
            end


        end
    end % methods

    %****************************************************
    %****************************************************
    methods(Static=true)
        %*************************************************************************************************************
        % Write RA and Dec data to CSV file.
        %
        % If <filename> exists then it is overwritten
        %
        % Inputs:
        %   filename    -- [str] name of CSV file
        %   ra          -- [float array(nDatums)] Returned Right Acsension
        %   dec         -- [float array(nDatums)] Returned Declination
        %   timestamps  -- [float array(nDatums)] timestamps in BTJD for row and col arrays above
        %
        % Outputs:
        %   CSV file named <filename>
        %   
        %*************************************************************************************************************
        function [] = write_ra_dec_to_csv(filename, ra, dec, timestamps) 

            if exist(filename, 'file')
                delete(filename)
            end

            for idx = 1:length(ra)
                row = [timestamps(idx), ra(idx), dec(idx)];
                dlmwrite(filename, row, '-append', 'precision', 7);
            end

        end

    end % methods

    %****************************************************
    %****************************************************
    methods(Access=private)
        
        %*************************************************************************************************************
        % Extracts the delta quaternions from a PA data object
        %
        % This code is taken from paDataClass.compute_target_centroids
        %
        % Outputs:
        %   obj.deltaQuaternions -- [struct array(nCadences)]
        %           .timestamps
        %           .cadenceNumbers
        %           .isValid
        %           .objects -- [quaternion class] (see quaternion.m)
        %
        %*************************************************************************************************************
        function extract_delta_quaternions(obj, paDataObject)

            N_CAMERAS = 4;
            N_ELEMENTS = 4;
            
            cadenceTimes = paDataObject.cadenceTimes;
            cadenceNumbers = cadenceTimes.cadenceNumbers;
            midTimestamps = cadenceTimes.midTimestamps;

            quaternionConfigurationMnemonics = paDataObject.quaternionAncillaryEngineeringConfigurationStruct.mnemonics;
            if (isempty(quaternionConfigurationMnemonics))
                print('Delta Quaternions are not avalable')
                obj.deltaQuaternions = [];
                return
            end
            
            ancillaryEngineeringDataStruct = paDataObject.ancillaryEngineeringDataStruct;
            
            quaternionEngineeringMnemonics = {ancillaryEngineeringDataStruct.mnemonic};
                
            quaternions = repmat(struct( ...
                'timestamps', [], ...
                'cadenceNumbers', [], ...
                'isValid', [], ...
                'objects', [] ), [1, N_CAMERAS]);
            
            rawQuaternions = quaternions;
            rawQuaternions(1).values = [];
    
            for iCamera = 1 : N_CAMERAS
                
                % Parse the delta quaternions.
                for iElement = 1 : N_ELEMENTS
            
                    mnemonicString = sprintf('cam%d_q%d', iCamera, iElement);
                    isMnemonic = contains(quaternionEngineeringMnemonics, mnemonicString);
                    if ~any(isMnemonic)
                        fprintf('Quaternions are not available for Camera %d (%d)\n', iCamera, iElement);
                        continue
                    end % if
            
                    if iElement == 1
                        quaternionTimestamps = ancillaryEngineeringDataStruct(isMnemonic).timestamps;
                        quaternionCadenceNumbers = ancillaryEngineeringDataStruct(isMnemonic).cadenceNumbers;
                        rawQuaternions(iCamera).timestamps = quaternionTimestamps;
                        rawQuaternions(iCamera).cadenceNumbers = quaternionCadenceNumbers;
                        rawQuaternions(iCamera).values = zeros(length(quaternionCadenceNumbers), N_ELEMENTS);     
                    end % if
            
                    values = ancillaryEngineeringDataStruct(isMnemonic).values;
                    rawQuaternions(iCamera).values(:, iElement) = values;
            
                end % for iElement
            
                if ~isempty(rawQuaternions(iCamera).timestamps)
                    quaternionsAvailable = true;
                end

              % % Perform the median detrending.
              % if ~isempty(rawQuaternions(iCamera).timestamps)
              %     timeIntervalDays = median(diff(rawQuaternions(iCamera).timestamps));
              %     filterOrder = ...
              %         round(FILTER_LENGTH_HOURS * DAYS_PER_HOUR / timeIntervalDays);
              %     filterOrder = 2 * fix(filterOrder/2) + 1;
              %     values = rawQuaternions(iCamera).values;
              %     values(:, 1:3) = values(:, 1:3) - medfilt1( ...
              %         values(:, 1:3), filterOrder, filterOrder, 1, 'omitnan');
              %     values(:, 4) = sqrt(1 - sum(values(:, 1:3).^2, 2));
              %     rawQuaternions(iCamera).values = values;
              % end % if
                        
                rawQuaternions(iCamera).isValid = all(~isnan(rawQuaternions(iCamera).values), 2);
                
                % Now build the struct that has one quaternion object per cadence.
                % Note that quaternion class ordering is [qr, qi, qj, qk] whereas
                % POC quaternion ordering is [qi, qj, qk, qr].
                uniqueCadences = unique(rawQuaternions(iCamera).cadenceNumbers);
                quaternions(iCamera).cadenceNumbers = uniqueCadences;
                quaternions(iCamera).timestamps = zeros(size(uniqueCadences));
                quaternions(iCamera).isValid = false(size(uniqueCadences));
                quaternions(iCamera).objects = repmat(quaternion([1, 0, 0, 0]), size(uniqueCadences));
                
                for iCadence = 1 : length(uniqueCadences)
                    cadenceNumber = uniqueCadences(iCadence);
                    timestamp = midTimestamps(cadenceNumbers == cadenceNumber);
                    validIndicators = rawQuaternions(iCamera).isValid & ...
                        rawQuaternions(iCamera).cadenceNumbers == cadenceNumber;
                    values = rawQuaternions(iCamera).values(validIndicators, :);
                    q = quaternion(values(:, [4, 1, 2, 3]));
                    quaternions(iCamera).cadenceNumbers(iCadence) = cadenceNumber;
                    quaternions(iCamera).timestamps(iCadence) = timestamp;
                    if any(validIndicators)
                        quaternions(iCamera).objects(iCadence) = q;
                        quaternions(iCamera).isValid(iCadence) = true;
                    end % if
                end % for iCadence
                
            end % for iCamera
            
            if ~quaternionsAvailable  
                quaternions = [];
            end % if

            obj.deltaQuaternions = quaternions;
     
        end % extract_delta_quaternions

    end % private methods

end % classdef paWrapperClass

