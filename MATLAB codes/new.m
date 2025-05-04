function motorCameraSimulation
    %% Global Variables (for timers, rotor animation, and logging)
    global scanTimer captureTimer rotorAngle
    scanTimer = [];
    captureTimer = [];
    rotorAngle = 0; % in radians

    %% Create GUI
    fig = uifigure('Name','Motor-Camera System Simulation','Position',[100,100,1000,600]);
    
    % Axes for simulated camera image
    axImage = uiaxes(fig, 'Position',[50,250,400,300]);
    title(axImage, 'Simulated Camera Image');
    
    % Axes for linear stage (vertical view)
    axStage = uiaxes(fig, 'Position',[500,250,150,300]);
    title(axStage, 'Linear Stage Position');
    xlabel(axStage, 'Stage');
    ylabel(axStage, 'Position (microns)');
    axStage.XLim = [0 1];
    axStage.YLim = [0 1000];
    hold(axStage, 'on');
    % Draw vertical stage line:
    plot(axStage, [0.5 0.5], [0 1000], 'k-', 'LineWidth',2);
    stageMarker = plot(axStage, 0.5, 0, 'ro', 'MarkerSize',12, 'MarkerFaceColor','r');
    
    % Axes for motor rotor animation
    axRotor = uiaxes(fig, 'Position',[50,50,100,100]);
    axis(axRotor, 'equal');
    axRotor.XLim = [-1.2 1.2];
    axRotor.YLim = [-1.2 1.2];
    title(axRotor, 'Motor Rotor');
    hold(axRotor, 'on');
    theta = linspace(0,2*pi,100);
    x_circle = cos(theta);
    y_circle = sin(theta);
    plot(axRotor, x_circle, y_circle, 'k-', 'LineWidth',2);  % Rotor circle
    rotorMarker = quiver(axRotor, 0, 0, 0.8, 0, 'r','LineWidth',2, 'MaxHeadSize',2);
    
    % Labels for UB, LB, and Encoder reading
    lblUB = uilabel(fig, 'Position',[700,500,250,30], 'Text','Upper Bound (UB): TBD');
    lblLB = uilabel(fig, 'Position',[700,460,250,30], 'Text','Lower Bound (LB): TBD');
    lblEncoder = uilabel(fig, 'Position',[700,420,250,30], 'Text','Encoder: 0.0 microns');
    
    % Buttons for scanning and capturing phases
    btnScan = uibutton(fig, 'push', 'Text','Start Scan',...
        'Position',[50,150,150,50], 'ButtonPushedFcn', @(~,~) startScan());
    btnCapture = uibutton(fig, 'push', 'Text','Start Capture',...
        'Position',[250,150,150,50], 'Enable','off', 'ButtonPushedFcn', @(~,~) startCapture());
    
    %% Simulation Parameters
    stagePositions = 0:10:1000;  % 0 to 1000 microns in 10-micron steps
    numPos = numel(stagePositions);
    % Use an advanced focus metric function
    focusMetricFun = @advancedFocusMetric;
    
    %% Declare UB and LB in the outer scope for later use in capture phase
    UB = NaN;
    LB = NaN;
    
    %% Data Logging Initialization
    logScan = zeros(numPos, 3);     % Columns: [Stage, FocusMetric, FuzzyOutput]
    maxCaptureSteps = numPos;       % Maximum possible capture steps
    logCapture = zeros(maxCaptureSteps, 5); % Columns: [CommandedPos, ActualPos, Error, Adjustment, FocusMetric]
    captureLogIndex = 1;
    
    %% Build Enhanced Focus FIS (FocusFIS) - Multi-level classification
    fis = mamfis('Name','FocusFIS');
    fis = addInput(fis, [0 100], 'Name', 'FocusMetric');
    % Three membership functions for focus metric:
    fis = addMF(fis, 'FocusMetric', 'trimf', [0 0 40], 'Name', 'Blurred');
    fis = addMF(fis, 'FocusMetric', 'trimf', [30 50 70], 'Name', 'Moderate');
    fis = addMF(fis, 'FocusMetric', 'trimf', [60 100 100], 'Name', 'Sharp');
    fis = addOutput(fis, [0 1], 'Name', 'FocusDegree');
    % Three membership functions for focus degree:
    fis = addMF(fis, 'FocusDegree', 'trimf', [0 0 0.3], 'Name', 'Low');
    fis = addMF(fis, 'FocusDegree', 'trimf', [0.2 0.5 0.8], 'Name', 'Medium');
    fis = addMF(fis, 'FocusDegree', 'trimf', [0.7 1 1], 'Name', 'High');
    % Fuzzy rules for focus evaluation:
    ruleListStr = 'If FocusMetric is Blurred then FocusDegree is Low'; 
    ruleListStr2 = 'If FocusMetric is Moderate then FocusDegree is Medium'; 
    ruleListStr3 = 'If FocusMetric is Sharp then FocusDegree is High'; 
    fis = addRule(fis, ruleListStr);
    fis = addRule(fis, ruleListStr2);
    fis = addRule(fis, ruleListStr3);
    focusThreshold = 0.5;  % Threshold on fuzzy output to determine in-focus region
    
    %% Build Enhanced Motor Regulation FIS (MotorFIS)
    motorFIS = mamfis('Name','MotorFIS');
    motorFIS = addInput(motorFIS, [-2 2], 'Name', 'Error');
    motorFIS = addMF(motorFIS, 'Error', 'trimf', [-2 -2 0], 'Name', 'Negative');
    motorFIS = addMF(motorFIS, 'Error', 'trimf', [-1 0 1], 'Name', 'Zero');
    motorFIS = addMF(motorFIS, 'Error', 'trimf', [0 2 2], 'Name', 'Positive');
    motorFIS = addOutput(motorFIS, [-2 2], 'Name', 'Adjustment');
    motorFIS = addMF(motorFIS, 'Adjustment', 'trimf', [-2 -2 0], 'Name', 'Increase');
    motorFIS = addMF(motorFIS, 'Adjustment', 'trimf', [-0.5 0 0.5], 'Name', 'NoChange');
    motorFIS = addMF(motorFIS, 'Adjustment', 'trimf', [0 2 2], 'Name', 'Decrease');
    % Basic fuzzy rules for motor control
    ruleListMotor = [1 1 1 1; 2 2 1 1; 3 3 1 1];
    motorFIS = addRule(motorFIS, ruleListMotor);
    
    %% Timer Objects for Simulation Phases
    if isempty(scanTimer) || ~isvalid(scanTimer)
        scanTimer = timer('ExecutionMode', 'fixedRate', 'Period', 0.2, 'TimerFcn', @scanStep);
    end
    if isempty(captureTimer) || ~isvalid(captureTimer)
        captureTimer = timer('ExecutionMode','fixedRate', 'Period', 0.5, 'TimerFcn', @captureStep);
    end
    
    %% Variables for Scanning and Capturing
    currentScanIndex = 1;
    currentCaptureIndex = 1;
    rotorAngle = 0; % Initialize rotor angle
    
    %% Scanning Phase Function
    function scanStep(~,~)
        if currentScanIndex <= numPos
            z = stagePositions(currentScanIndex);
            % Generate a synthetic image with texture and blur based on z
            img = generateSyntheticImage(z);
            % Compute advanced focus measure (Variance of Laplacian)
            focusVal = computeVarianceOfLaplacian(img);
            % Scale focusVal to [0,100] for FIS input
            focusValScaled = min(max(focusVal,0),100);
            
            % Save focus values
            logScan(currentScanIndex, :) = [z, focusValScaled, evalfis(fis, focusValScaled)];
            
            % Update simulated camera image
            imshow(img, 'Parent', axImage);
            title(axImage, sprintf('Scan: Z = %d microns, Focus = %.1f', z, focusValScaled));
            
            % Update encoder and stage marker
            lblEncoder.Text = sprintf('Encoder: %d microns', z);
            stageMarker.YData = z;
            
            % Plot focus metric vs. position in real time
            hold(axStage, 'on');
            plot(axStage, z, focusValScaled, 'b*');
            
            currentScanIndex = currentScanIndex + 1;
        else
            % Determine in-focus region (Upper Bound and Lower Bound) based on fuzzy output threshold
            fuzzyOut = logScan(:,3);
            idxInFocus = find(fuzzyOut > focusThreshold);
            if ~isempty(idxInFocus)
                UB = stagePositions(idxInFocus(1));
                LB = stagePositions(idxInFocus(end));
            end
            lblUB.Text = sprintf('Upper Bound (UB): %.1f microns', UB);
            lblLB.Text = sprintf('Lower Bound (LB): %.1f microns', LB);
            stop(scanTimer);
            if ~isnan(UB) && ~isnan(LB)
                btnCapture.Enable = 'on';
            else
                uialert(fig, 'No in-focus region detected during scan!', 'Scan Error');
            end
            % Save scan log data to CSV
            writematrix(logScan(1:currentScanIndex-1, :), fullfile(pwd, 'scanLog.csv'));
        end
    end
    
    %% Capture Phase Function: Motor Regulation, Rotor Animation, Vibration & Closed-loop Control
    function captureStep(~,~)
        if currentCaptureIndex == 1
            [~, idx] = min(abs(stagePositions - UB));
            currentCaptureIndex = idx;
        end
        if currentCaptureIndex < numPos && stagePositions(currentCaptureIndex) < LB
            commandedStep = 10;
            commandedPos = stagePositions(currentCaptureIndex);
            % Simulate actual motor error with vibration/noise
            randomError = -2 + 4*rand() + 0.5*sin(2*pi*rand());
            actualPos = commandedPos + commandedStep + randomError;
            errorVal = (actualPos - commandedPos) - commandedStep;
            adjustment = evalfis(motorFIS, errorVal);
            % Dynamic velocity control based on focus sharpness
            % Use last scanned focus value as proxy for current focus quality
            lastFocus = logScan(max(currentScanIndex-1,1),2)/100;  % normalized [0,1]
            velocityFactor = 1 + (1 - lastFocus)*0.5;  
            adjustedStep = (commandedStep + adjustment) * velocityFactor;
            commandedPosNext = commandedPos + adjustedStep;
            
            % Update encoder and stage marker
            lblEncoder.Text = sprintf('Encoder: %.1f microns (Error: %.2f, Adj: %.2f)', actualPos, errorVal, adjustment);
            stageMarker.YData = actualPos;
            
            % Compute focus at actual position
            img = generateSyntheticImage(actualPos);
            focusVal = computeVarianceOfLaplacian(img);
            focusValScaled = min(max(focusVal,0),100);
            
            % Log capture data
            logCapture(captureLogIndex, :) = [commandedPos, actualPos, errorVal, adjustment, focusValScaled];
            captureLogIndex = captureLogIndex + 1;
            
            % Update simulated camera image
            imshow(img, 'Parent', axImage);
            title(axImage, sprintf('Capture: Z = %.1f microns, Focus = %.1f', actualPos, focusValScaled));
            
            % Update motor rotor animation
            rotorAngle = rotorAngle + (5*pi/180);
            set(rotorMarker, 'UData', 0.8*cos(rotorAngle), 'VData', 0.8*sin(rotorAngle));
            
            % Determine next stage index
            [~, nextIdx] = min(abs(stagePositions - commandedPosNext));
            currentCaptureIndex = nextIdx;
        else
            stop(captureTimer);
            uialert(fig, 'Capture phase complete!', 'Done');
            % Save capture log data to CSV
            writematrix(logCapture(1:captureLogIndex-1, :), fullfile(pwd, 'captureLog.csv'));
            % Display and save FIS graphs and rules
            displayAndSaveFIS(fis, motorFIS);
        end
    end
    
    %% Button Callback: Start Scan
    function startScan()
        currentScanIndex = 1;
        logScan = zeros(numPos, 3);
        logCapture = zeros(numPos, 5);
        start(scanTimer);
        btnScan.Enable = 'off';
    end
    
    %% Button Callback: Start Capture
    function startCapture()
        currentCaptureIndex = 1;
        start(captureTimer);
        btnCapture.Enable = 'off';
    end
end

%% Helper Function to Display and Save FIS Membership Functions and Rules
function displayAndSaveFIS(fis, motorFIS)
    % Create visualizations folder if it doesn't exist
    visFolder = fullfile(pwd, 'visualizations');
    if ~exist(visFolder, 'dir')
        mkdir(visFolder);
    end

    % Display FocusFIS membership functions
    figFocus = figure('Name','FocusFIS Membership Functions');
    subplot(2,1,1);
    plotmf(fis, 'input', 1);
    title('FocusFIS - Input: FocusMetric');
    subplot(2,1,2);
    plotmf(fis, 'output', 1);
    title('FocusFIS - Output: FocusDegree');
    saveas(figFocus, fullfile(visFolder, 'FocusFIS_MF.png'));
    disp('FocusFIS Rules:');
    disp(fis.Rules);
    
    % Display MotorFIS membership functions
    figMotor = figure('Name','MotorFIS Membership Functions');
    subplot(2,1,1);
    plotmf(motorFIS, 'input', 1);
    title('MotorFIS - Input: Error');
    subplot(2,1,2);
    plotmf(motorFIS, 'output', 1);
    title('MotorFIS - Output: Adjustment');
    saveas(figMotor, fullfile(visFolder, 'MotorFIS_MF.png'));
    disp('MotorFIS Rules:');
    disp(motorFIS.Rules);
end

%% Advanced Focus Detection Functions
function img = generateSyntheticImage(z)
    % Generate a synthetic textured image whose sharpness depends on stage position z.
    % Optimal focus is around 400 microns; blur increases with distance from 400.
    baseImg = uint8(255*rand(300,400));
    sigma = abs(z-400)/200;  % Adjust blur level based on distance from 400
    if sigma > 0.1
        img = imgaussfilt(baseImg, sigma);
    else
        img = baseImg;
    end
end

function focusVal = computeVarianceOfLaplacian(img)
    % Compute the variance of the Laplacian of the image (a measure of focus/sharpness)
    laplacianKernel = [0 1 0; 1 -4 1; 0 1 0];
    lapImg = imfilter(double(img), laplacianKernel, 'replicate');
    focusVal = var(lapImg(:));
end

function fm = advancedFocusMetric(z)
    % Advanced focus metric: generate an image and compute its focus measure
    img = generateSyntheticImage(z);
    fm = computeVarianceOfLaplacian(img);
end
