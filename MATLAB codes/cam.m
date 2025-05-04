function cameraEdgeMotorSimulation
    % This script uses a REAL camera feed (via Image Acquisition Toolbox)
    % to detect edges and compute a focus measure. The user physically adjusts
    % the camera's focus. We simulate the linear stage movement, fuzzy logic
    % for focus detection, and motor regulation (for nominal 10-micron steps).

    %% Global Timers and Rotor Angle
    global scanTimer captureTimer rotorAngle
    scanTimer = [];
    captureTimer = [];
    rotorAngle = 0; % in radians

    %% --- Create a Live Camera Object ---
    % Adjust 'winvideo' and '1' if your camera is different.
    vid = videoinput('winvideo', 1);
    src = getselectedsource(vid);
    % Optionally set camera properties here, e.g.:
    % src.ExposureMode = 'manual'; src.Exposure = -5;
    % start(vid);  % We do snapshot() calls below, so we don't necessarily need start().

    %% --- Create the Main GUI Figure ---
    fig = uifigure('Name','Motor-Camera System with Live Edge Detection','Position',[100,100,1000,600]);

    % Axes for the live camera feed
    axImage = uiaxes(fig, 'Position',[50,250,400,300]);
    title(axImage, 'Live Camera Feed');

    % Axes for the linear stage
    axStage = uiaxes(fig, 'Position',[500,250,150,300]);
    title(axStage, 'Linear Stage');
    xlabel(axStage, 'Stage');
    ylabel(axStage, 'Position (microns)');
    axStage.XLim = [0 1];
    axStage.YLim = [0 1000];
    hold(axStage, 'on');
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
    plot(axRotor, x_circle, y_circle, 'k-', 'LineWidth',2); % rotor circle
    rotorMarker = quiver(axRotor, 0, 0, 0.8, 0, 'r','LineWidth',2, 'MaxHeadSize',2);

    % Labels for upper bound (UB), lower bound (LB), and encoder
    lblUB = uilabel(fig, 'Position',[700,500,250,30], 'Text','Upper Bound (UB): TBD');
    lblLB = uilabel(fig, 'Position',[700,460,250,30], 'Text','Lower Bound (LB): TBD');
    lblEncoder = uilabel(fig, 'Position',[700,420,250,30], 'Text','Encoder: 0.0 microns');

    % Buttons for scanning and capturing phases
    btnScan = uibutton(fig, 'push', 'Text','Start Scan',...
        'Position',[50,150,150,50], 'ButtonPushedFcn', @(~,~) startScan());
    btnCapture = uibutton(fig, 'push', 'Text','Start Capture',...
        'Position',[250,150,150,50], 'Enable','off', 'ButtonPushedFcn', @(~,~) startCapture());

    %% --- Simulation Parameters ---
    stagePositions = 0:10:100;   % 0 to 1000 microns in 10-micron steps
    numPos = numel(stagePositions);

    % We'll define a function to measure focus by counting edges:
    % Convert image to grayscale, run Canny edge, and sum up the edge pixels.
    % We'll scale it so the max is ~100 for fuzzy logic.
    function focusVal = computeFocusFromEdges(frame)
        gray = rgb2gray(frame);
        edges = edge(gray, 'Canny');
        rawCount = sum(edges(:));  % # of edge pixels
        % Scale rawCount to [0..100] range (tweak as needed)
        focusVal = min(rawCount / 2000, 1) * 100; % Adjust the divisor '2000' for your camera
    end

    %% --- Build FIS for Focus Determination (FocusFIS) ---
    fis = mamfis('Name','FocusFIS');
    fis = addInput(fis, [0 100], 'Name', 'FocusMetric');
    fis = addMF(fis, 'FocusMetric', 'trimf', [0 0 50], 'Name', 'NotFocused');
    fis = addMF(fis, 'FocusMetric', 'trimf', [30 100 100], 'Name', 'Focused');
    fis = addOutput(fis, [0 1], 'Name', 'FocusDegree');
    fis = addMF(fis, 'FocusDegree', 'trimf', [0 0 0.5], 'Name', 'NotFocus');
    fis = addMF(fis, 'FocusDegree', 'trimf', [0.5 1 1], 'Name', 'Focus');
    ruleList = "FocusMetric==Focused => FocusDegree=Focus";
    fis = addRule(fis, ruleList);
    focusThreshold = 0.5;

    %% --- Motor Regulation FIS (MotorFIS) ---
    motorFIS = mamfis('Name','MotorFIS');
    motorFIS = addInput(motorFIS, [-2 2], 'Name', 'Error');
    motorFIS = addMF(motorFIS, 'Error', 'trimf', [-2 -2 0], 'Name', 'Negative');
    motorFIS = addMF(motorFIS, 'Error', 'trimf', [-1 0 1], 'Name', 'Zero');
    motorFIS = addMF(motorFIS, 'Error', 'trimf', [0 2 2], 'Name', 'Positive');
    motorFIS = addOutput(motorFIS, [-2 2], 'Name', 'Adjustment');
    motorFIS = addMF(motorFIS, 'Adjustment', 'trimf', [-2 -2 0], 'Name', 'Increase');
    motorFIS = addMF(motorFIS, 'Adjustment', 'trimf', [-0.5 0 0.5], 'Name', 'NoChange');
    motorFIS = addMF(motorFIS, 'Adjustment', 'trimf', [0 2 2], 'Name', 'Decrease');
    ruleListMotor = [1 1 1 1; 2 2 1 1; 3 3 1 1];
    motorFIS = addRule(motorFIS, ruleListMotor);

    %% Variables for scanning and capturing
    focusValues = zeros(1, numPos);
    fuzzyOut = zeros(1, numPos);
    UB = NaN; LB = NaN;
    currentScanIndex = 1;
    currentCaptureIndex = 1;
    rotorAngle = 0;

    %% Timers for scanning/capturing
    global scanTimer captureTimer
    if isempty(scanTimer) || ~isvalid(scanTimer)
        scanTimer = timer('ExecutionMode', 'fixedRate', 'Period', 0.2, 'TimerFcn', @scanStep);
    end
    if isempty(captureTimer) || ~isvalid(captureTimer)
        captureTimer = timer('ExecutionMode','fixedRate', 'Period', 0.5, 'TimerFcn', @captureStep);
    end

    %% --- Scanning Phase ---
    function scanStep(~,~)
        if currentScanIndex <= numPos
            z = stagePositions(currentScanIndex);
            % Acquire a frame from the real camera
            frame = getsnapshot(vid);
            % Display it
            imshow(frame, 'Parent', axImage);
            title(axImage, sprintf('Scan: Position = %d microns', z));

            % Compute a focus measure from edges
            focusVal = computeFocusFromEdges(frame);
            focusValues(currentScanIndex) = focusVal;
            fuzzyVal = evalfis(fis, focusVal);
            fuzzyOut(currentScanIndex) = fuzzyVal;

            % Update stage marker and encoder
            lblEncoder.Text = sprintf('Encoder: %d microns', z);
            stageMarker.YData = z;

            currentScanIndex = currentScanIndex + 1;
        else
            % Scanning done, find UB/LB
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
                uialert(fig, 'No in-focus region detected!', 'Scan Error');
            end
        end
    end

    %% --- Capture Phase (Motor Regulation + Rotor Animation) ---
    function captureStep(~,~)
        if currentCaptureIndex == 1
            [~, idx] = min(abs(stagePositions - UB));
            currentCaptureIndex = idx;
        end
        if currentCaptureIndex < numPos && stagePositions(currentCaptureIndex) < LB
            commandedPos = stagePositions(currentCaptureIndex);
            commandedStep = 10;

            % Simulate actual motor error
            randomError = -2 + 4*rand();
            actualPos = commandedPos + commandedStep + randomError;
            errorVal = (actualPos - commandedPos) - commandedStep;
            adjustment = evalfis(motorFIS, errorVal);
            adjustedStep = commandedStep + adjustment;
            commandedPosNext = commandedPos + adjustedStep;

            % Acquire a frame from camera
            frame = getsnapshot(vid);
            imshow(frame, 'Parent', axImage);
            focusVal = computeFocusFromEdges(frame);
            title(axImage, sprintf('Capture: Z=%.1f microns, Focus=%.1f', actualPos, focusVal));

            % Update encoder/stage
            lblEncoder.Text = sprintf('Encoder: %.1f microns (Err=%.2f, Adj=%.2f)', actualPos, errorVal, adjustment);
            stageMarker.YData = actualPos;

            % Rotate the motor rotor by 5 degrees
            rotorAngle = rotorAngle + 5*pi/180;
            updateRotor(rotorMarker, rotorAngle);

            % Next index
            [~, nextIdx] = min(abs(stagePositions - commandedPosNext));
            currentCaptureIndex = nextIdx;
        else
            stop(captureTimer);
            uialert(fig, 'Capture phase complete!', 'Done');
            % Optionally display FIS info
            displayAndSaveFIS(fis, motorFIS);
        end
    end

    %% Start Scan Callback
    function startScan()
        currentScanIndex = 1;
        focusValues = zeros(1, numPos);
        fuzzyOut = zeros(1, numPos);
        start(scanTimer);
        btnScan.Enable = 'off';
    end

    %% Start Capture Callback
    function startCapture()
        currentCaptureIndex = 1;
        start(captureTimer);
        btnCapture.Enable = 'off';
    end

end

%% Helper: Rotate the rotor arrow in axRotor
function updateRotor(rotorMarker, angle)
    set(rotorMarker, 'UData', 0.8*cos(angle), 'VData', 0.8*sin(angle));
end

%% Helper: Display and Save FIS membership functions
function displayAndSaveFIS(fis, motorFIS)
    figFocus = figure('Name','FocusFIS Memberships');
    subplot(2,1,1);
    plotmf(fis, 'input', 1);
    title('FocusFIS - Input');
    subplot(2,1,2);
    plotmf(fis, 'output', 1);
    title('FocusFIS - Output');
    saveas(figFocus, fullfile(pwd, 'FocusFIS_MF.png'));
    disp('FocusFIS Rules:');
    disp(fis.Rules);

    figMotor = figure('Name','MotorFIS Memberships');
    subplot(2,1,1);
    plotmf(motorFIS, 'input', 1);
    title('MotorFIS - Error');
    subplot(2,1,2);
    plotmf(motorFIS, 'output', 1);
    title('MotorFIS - Adjustment');
    saveas(figMotor, fullfile(pwd, 'MotorFIS_MF.png'));
    disp('MotorFIS Rules:');
    disp(motorFIS.Rules);
end

