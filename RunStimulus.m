% Start matlab with ptb3-matlab DriftDemo.m
try
    Screen('Preference', 'SkipSyncTests', 1);
    Screen('Preference', 'VisualDebugLevel', 3);

    rng(1, "twister"); % random seed

    AssertOpenGL;

    % Get the list of screens and choose the one with the highest screen number.
    screens = Screen('Screens');
    screenNumber = max(screens);

    % Find the color values for white and black:
    white = WhiteIndex(screenNumber);
    black = BlackIndex(screenNumber);
    gray = round((white + black) / 2);
    if gray == white
        gray = white / 2;
    end
    inc = white - gray;

    % Open a window with a gray background color:
    [w, windowRect] = Screen('OpenWindow', screenNumber, gray);

    %% Windowed Version %%
    % windowWidth = 1920;
    % windowHeight = 1080;
    % 
    % % Get full screen size to compute center
    % screenRes = Screen('Rect', screenNumber);
    % screenCenterX = (screenRes(3) - screenRes(1)) / 2;
    % screenCenterY = (screenRes(4) - screenRes(2)) / 2;
    % 
    % % Compute upper-left and lower-right corner of centered window
    % left = round(screenCenterX - windowWidth / 2);
    % top = round(screenCenterY - windowHeight / 2);
    % right = left + windowWidth;
    % bottom = top + windowHeight;
    % 
    % % Define the window rectangle and open it
    % customRect = [left, top, right, bottom];
    % [w, windowRect] = Screen('OpenWindow', screenNumber, gray, customRect);
    %% Windowed Version %%


    % Get screen dimensions dynamically
    [screenXpixels, screenYpixels] = Screen('WindowSize', w);
    [xCenter, yCenter] = RectCenter(windowRect);

    % Compute dynamic grid size based on screen resolution
    [x, y] = meshgrid(-screenXpixels/2:screenXpixels/2, -screenYpixels/2:screenYpixels/2);

    % Grating parameters
    numFrames = 24; % Temporal period in frames
    angles = [0 45 90 135 180 225 270 315];
    f = 0.005 * 2 * pi; % Cycles/pixel42
    tex = zeros(length(angles), numFrames);

    % Create textures dynamically
    for k = 1:length(angles)
        angle = angles(k) * pi / 180; % Convert to radians
        for i = 1:numFrames
            phase = (i / numFrames) * 2 * pi;
            a = cos(angle) * f;
            b = sin(angle) * f;
            m = sin(a * x + b * y + phase);
            m(m > 0) = 1;
            m(m <= 0) = -1;
            tex(k, i) = Screen('MakeTexture', w, round(gray + inc * m));
        end
    end

    % Run the animation
    movieDurationSecs = 0.5;
    ITIDurationSecs = 0.5;
    frameRate = Screen('FrameRate', screenNumber);
    if frameRate == 0
        frameRate = 60; % Assume 60 Hz if unknown
    end

    movieDurationFrames = round(movieDurationSecs * frameRate);
    movieFrameIndices = mod(0:(movieDurationFrames-1), numFrames) + 1;
    ITIDurationFrames = round(ITIDurationSecs * frameRate);

    % PseudoRandom Design
    blockdesign = [];
    numblocks = 5; % Trials per stimulus type, each is 8s
    for i = 1:numblocks
        blockdesign = [blockdesign randperm(length(angles))];
    end
    numtrials = length(angles) * numblocks;
    
    % Save stimulus directions to a text file
    stimulusDirections = angles(blockdesign); % Convert blockdesign to actual angles
    
    fileID = fopen('StimulusDirections.txt', 'w');
    fprintf(fileID, 'Trial\tDirection (Degrees)\n');
    for trial = 1:length(stimulusDirections)
        fprintf(fileID, '%d\t%d\n', trial, stimulusDirections(trial));
    end
    fclose(fileID);
    
    % Save in MATLAB format
    save('StimulusDirections.mat', 'stimulusDirections');

    % Make an example GIF from the textures (one angle only, e.g., 90°)
    gifAngleIdx = find(angles == 90);  % Change angle as needed
    gifFilename = 'grating_animation_90deg.gif';
    
    for i = 1:numFrames
        % Get image matrix from texture
        imgMatrix = Screen('GetImage', tex(gifAngleIdx, i));
    
        % Convert to indexed image for GIF
        [A, map] = rgb2ind(imgMatrix, 256);
        
        % Write to GIF file
        if i == 1
            imwrite(A, map, gifFilename, 'gif', 'LoopCount', Inf, 'DelayTime', 1/frameRate);
        else
            imwrite(A, map, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', 1/frameRate);
        end
    end


    
    % Run trials
    for k = 1:numtrials      
        for i = 1:movieDurationFrames
            % Check for key press to exit
            [keyIsDown, ~, keyCode] = KbCheck;
            if keyIsDown && keyCode(KbName('ESCAPE'))  % Exit on 'q' key
                sca;  % Close the window
                return;
            end
            
            % Draw image:
            Screen('DrawTexture', w, tex(blockdesign(k),movieFrameIndices(i)));
            % Draws white box
            Screen('FillRect', w,[255 255 255],[0 0 200 200]); % Photodiode indicator
            Screen('Flip', w);
        end
    
        % Inter-trial interval
        for i = 1:ITIDurationFrames
            [keyIsDown, ~, keyCode] = KbCheck;
            if keyIsDown && keyCode(KbName('ESCAPE'))  % Check again during ITI
                sca;
                return;
            end
            Screen('Flip', w);
        end
    end

    % Clean up
    Priority(0);
    Screen('Close');
    sca;
    save Stimlog blockdesign angles;

catch
    % Catch errors and close gracefully
    sca;
    Priority(0);
    psychrethrow(psychlasterror);
end


