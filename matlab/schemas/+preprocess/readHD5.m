function [data, settings, version] = readHD5(F)
% F: Key structure or full filename including path (for example '/path/Patchfile10.h5')
%
% data: all variables extracted from file
% settings: settings telegraphs from NPI amp
% ver: file version

if isstruct(F)
    if count(experiment.ScanBehaviorFile & F)
        F = fullfile(getLocalPath(fetch1(experiment.Session & F,'behavior_path')), ...
            fetch1(experiment.ScanBehaviorFile & F, 'filename'));
    else
        filename = fetch1(experiment.Scan & F,'filename');
        filename = strsplit(filename, '_');
        scanId = str2double(filename{2});
        filename = filename{1};
        F = fullfile( ...
            getLocalPath(fetch1(experiment.Session & F,'behavior_path')), ...
            sprintf('%s%d0.h5', filename, scanId));
    end
end

% Check that file exists
F = strrep(F,'%d','0'); % FIX trailing %d.h5 not found error
f = dir(F);
assert(length(f)==1,['Cannot find file ' F]);
F = [F(1:end-4) '%d.h5']; % Append '%d.h5' to filename in place of trailing 'x.h5'

try
    % open file using family driver
    fapl = H5P.create('H5P_FILE_ACCESS');
    H5P.set_fapl_family(fapl,2^31,'H5P_DEFAULT');
    fp = H5F.open(F,'H5F_ACC_RDONLY',fapl);
    H5P.close(fapl);
catch
    error 'File version not known'
end

% Get version of the file
version  = H5Tools.readAttribute(fp,'Version');

switch version
    case 1
        % Files recorded after 03-28-2014 using the NPI ELC-03XS amplifier as amp 1 and the AxoClamp 2B (.1x headstage) as amp 2
        % Amp 2 current low-pass is set to 3000Hz.
        % Skips settings telegraph from NPI if any(~sets), i.e. if NPI amp is turned off.
        % Adds scanimage sync channel
        % Ball data is from optical encoder
        % ts2sec now takes 'packetLen' argument in order to correctly assign timestamps to end of data packets. This version adds a analogPacketLen field to data struct
        
        % *** Packet length is set at 2000 for analog channels ***
        ANALOG_PACKET_LEN = 2000;
        
        %data1d = H5Tools.readDataset(fp,'dataset1d')
        data.ball = H5Tools.readDataset(fp,'ball') ;
        wf = H5Tools.readDataset(fp,'waveform') ;
        sets = H5Tools.readDataset(fp,'settings') ;
        data.cam1ts = H5Tools.readDataset(fp,'behaviorvideotimestamp') ;
        data.cam2ts = H5Tools.readDataset(fp,'eyetrackingvideotimestamp') ;
        
        waveformDescStr=H5Tools.readAttribute(fp,'waveform Channels Description')';
        assert(strcmp(deblank(waveformDescStr),'Current Input 1, Voltage Input 1, Sync Photodiode, Stimulation Photodiode, LED Level Input, Patch Command Input, Shutter, Current Input 2, Voltage Input 2, Scan Image Sync'),...
            'waveform Channels Description is wrong for this file version');
        
        settingsDescStr=H5Tools.readAttribute(fp,'settings Channels Description')';
        assert(strcmp(deblank(settingsDescStr),'Current Gain, Voltage Gain, Current Low Pass, Voltage Low Pass, Voltage High Pass'),...
            'settings Channels Description is wrong for this file version');
        
        %convert waveform to structure
        data.i1 = wf(:,1);
        data.v1 = wf(:,2);
        data.i2 = wf(:,8);
        data.v2 = wf(:,9);
        data.syncPd = wf(:,3);
        data.stimPd = wf(:,4);
        data.led = wf(:,5);
        data.command = wf(:,6);
        data.shutter = wf(:,7);
        data.scanImage = wf(:,10);
        data.ts = wf(:,11);
        data.analogPacketLen = ANALOG_PACKET_LEN;
        
        if ~any(~round(sets(:)))
            % deal with setting telegraphs on NPI amp
            iGains = [0.1 0.2 0.5 1 2 5 10];
            settings.iGain = iGains(unique(round(sets(:,1))));
            assert(length(settings.iGain)==1,'Current gain changed during recording');
            
            vGains = [10 20 50 100 200 500 1000];
            settings.vGain = vGains(unique(round(sets(:,2))));
            assert(length(settings.vGain)==1,'Voltage gain changed during recording');
            
            iLowPassCorners = [20 50 100 200 300 500 700 1000 1300 2000 3000 5000 8000 10000 13000 20000];
            settings.iLowPass = iLowPassCorners(unique(round(sets(:,3)))+9);
            assert(length(settings.iLowPass)==1,'Current low pass filter changed during recording');
            
            vLowPassCorners = [20 50 100 200 300 500 700 1000 1300 2000 3000 5000 8000 10000 13000 20000];
            settings.vLowPass = vLowPassCorners(unique(round(sets(:,4)))+9);
            assert(length(settings.vLowPass)==1,'Voltage low pass filter changed during recording');
            
            vHighPassCorners = [0 0.1 0.3 0.5 1 3 5 10 30 50 100 300 500 800 1000 3000];
            settings.vHighPass = vHighPassCorners(unique(round(sets(:,5)))+9);
            assert(length(settings.vHighPass)==1,'Voltage high pass filter changed during recording');
        else
            % constant settings on unused NPI amp
            
            settings.iGain = 1;
            settings.vGain = 1;
            settings.iLowPass = 1;
            settings.vLowPass = 1;
            settings.vHighPass = 1;
            %warning('Unable to read settings telegraphs from NPI Amp');
        end
        
        
        % settings on AxoClamp 2B are constant
        settings(2).iGain = 0.1;
        settings(2).vGain = 10;
        settings(2).vLowPass = 30000;
        settings(2).iLowPass = 3000;
        settings(2).vHighPass = 0;
        
        % apply gains to voltage and current
        data.v1 = data.v1/settings(1).vGain;
        data.i1 = data.i1/settings(1).iGain;
        data.v2 = data.v2/settings(2).vGain;
        data.i2 = data.i2/settings(2).iGain;
        
        
        H5F.close(fp);
        
    case {2, 2.1}
        % WH_channelNames = Position,Counter,Time
        % IM_channelNames = CTR Time, Real Time

        % Packet Length
        data.analogPacketLen  = (H5Tools.readAttribute(fp,'AS_samples_per_channel'));
        
        % read wheel trace
        data.ball = H5Tools.readDataset(fp,'Wheel') ;
        
        % read camera timestamps
        data.eyecam_ts = H5Tools.readDataset(fp,'videotimestamps') ;
        
        % read frame timestamps
        data.framenum_ts = H5Tools.readDataset(fp,'framenum_ts') ;
        
        % read trial timestamps
        data.trialnum_ts = H5Tools.readDataset(fp,'trialnum_ts') ;
        
        % check for correct waveform structure
        waveformDescStr=H5Tools.readAttribute(fp,'AS_channelNames')';
        
        % read wf and convert waveform to structure
        asVersion = H5Tools.readAttribute(fp,'AS_Version');
        switch asVersion
            case 2
                idx1 = find(waveformDescStr == ',') ; % need to take care of a comma misplacement in some files
                waveformDescStr(idx1) = ' ' ;
                teststr = 'Photodiode, FrameSync, Time' ;
                idx2 = find(teststr == ',') ;
                teststr(idx2) = ' ' ;
                assert(strcmp(deblank(waveformDescStr),deblank(teststr)),...
            'waveform Channels Description is wrong for this file version');
                wf = H5Tools.readDataset(fp,'Analog Signals') ;
                data.syncPd = wf(:,1);
                data.scanImage = wf(:,2);
                data.ts = wf(:,3);
            case 2.1
                wfstr = deblank(waveformDescStr) ;
                assert(strcmp(wfstr,'Photodiode, ScanImageFrameSync, LaserPower ,Time') ||...
                    strcmp(wfstr,'Photodiode, ScanImageFrameSync, LaserPower, Time') ||...
                    strcmp(wfstr,'Photodiode, ScanImageFrameSync, LaserPower, AudioStimWindow, Time'),...
            'waveform Channels Description is wrong for this file version');
                idx = strfind(wfstr,',') ;
                ch_names = {} ;
                for ii=1:length(idx)+1
                    if ii == 1
                        ch_names{ii} = sscanf(wfstr(1:idx(ii)-1),'%s') ;
                    elseif ii == length(idx)+1
                        ch_names{ii} = sscanf(wfstr(idx(ii-1)+1:end),'%s') ;
                    else
                        ch_names{ii} = sscanf(wfstr(idx(ii-1)+1:idx(ii)-1),'%s') ;
                    end
                end                        
                wf = H5Tools.readDataset(fp,'Analog Signals') ;
                ipos = strcmp(ch_names, 'Photodiode') ;
                data.syncPd = wf(:,ipos);
                ipos = strcmp(ch_names, 'ScanImageFrameSync') ;
                data.scanImage = wf(:,ipos);
                ipos = strcmp(ch_names, 'Time') ;
                data.ts = wf(:,ipos);
            case 2.2
                assert(strcmp(deblank(waveformDescStr),'Photodiode, ScanImageFrameSync, LaserPower, Temperature ,Time'),...
            'waveform Channels Description is wrong for this file version');
                wf = H5Tools.readDataset(fp,'Analog Signals') ;
                data.syncPd = wf(:,1);
                data.scanImage = wf(:,2);
                data.ts = wf(:,5);        
        end
                
        % close file
        H5F.close(fp);
        
    otherwise
        error 'File version not known'
end
