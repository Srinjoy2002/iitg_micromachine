import sys, os, math, random, time, queue, threading
import numpy as np, cv2
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import traceback


def computeVarianceOfLaplacian(img):
    """
    Compute the variance of the Laplacian of the image (a measure of focus/sharpness).
    """
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return lap.var()

def advancedFocusMetric(gray):
    """
    Compute focus measure on a grayscale image.
    """
    return computeVarianceOfLaplacian(gray)


class ScanThread(threading.Thread):
    def __init__(self, stagePositions, focusThreshold, focus_ctrl, cap, cam_lock):
        super().__init__()
        self.stagePositions = stagePositions
        self.numPos = len(stagePositions)
        self.focusThreshold = focusThreshold
        self.focus_ctrl = focus_ctrl  # yh hai control system algo
        self.cap = cap
        self.cam_lock = cam_lock
        self.logScan = np.zeros((self.numPos, 3))  # [Stage, FocusMetric, FuzzyOutput]
        self.queue = queue.Queue()
    def run(self):
        try:
           
            local_focus_sim = ctrl.ControlSystemSimulation(self.focus_ctrl)
            for i, z in enumerate(self.stagePositions):
                with self.cam_lock:
                    ret, frame = self.cap.read()
                if not ret:
                    print("ScanThread: Unable to read frame")
                    time.sleep(0.2)
                    continue
                # Convert the captured frame (BGR) to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                focusVal = advancedFocusMetric(gray)
                focusValScaled = np.clip(focusVal, 0, 100)
                local_focus_sim.input['FocusMetric'] = focusValScaled
                local_focus_sim.compute()
                fuzzyOutput = local_focus_sim.output['FocusDegree']
                self.logScan[i, :] = [z, focusValScaled, fuzzyOutput]
                # Put update data into the queue
                self.queue.put(('update', z, focusValScaled, fuzzyOutput, gray))
                time.sleep(0.2)
            self.queue.put(('done',))
        except Exception as e:
            print("Exception in ScanThread:")
            traceback.print_exc()
            self.queue.put(('done',))

class CaptureThread(threading.Thread):
    def __init__(self, stagePositions, UB, LB, motor_ctrl, logScan, cap, cam_lock):
        super().__init__()
        self.stagePositions = stagePositions
        self.numPos = len(stagePositions)
        self.UB = UB
        self.LB = LB
        self.motor_ctrl = motor_ctrl 
        self.logScan = logScan      # from scan phase (for focus quality)
        self.cap = cap
        self.cam_lock = cam_lock
        self.captureLog = np.zeros((self.numPos, 5))  # [CommandedPos, ActualPos, Error, Adjustment, FocusMetric]
        self.queue = queue.Queue()
        self.currentCaptureIndex = 0
        self.rotorAngle = 0
    def run(self):
        try:
            local_motor_sim = ctrl.ControlSystemSimulation(self.motor_ctrl)
            # Set starting index based on UB
            idx = int(np.argmin(np.abs(self.stagePositions - self.UB)))
            self.currentCaptureIndex = idx
            while (self.currentCaptureIndex < self.numPos and 
                   self.stagePositions[self.currentCaptureIndex] < self.LB):
                commandedStep = 100       #100 micron step size
                commandedPos = self.stagePositions[self.currentCaptureIndex]
                randomError = -2 + 4 * random.random() + 0.5 * math.sin(2 * math.pi * random.random())
                actualPos = commandedPos + commandedStep + randomError
                errorVal = (actualPos - commandedPos) - commandedStep
                local_motor_sim.input['Error'] = errorVal
                local_motor_sim.compute()
                adjustment = local_motor_sim.output['Adjustment']
                # Use last scanned focus value (normalized) as a proxy for focus quality
                lastFocus = self.logScan[max(self.currentCaptureIndex-1, 0), 1] / 100.0
                velocityFactor = 1 + (1 - lastFocus) * 0.5
                adjustedStep = (commandedStep + adjustment) * velocityFactor
                commandedPosNext = commandedPos + adjustedStep
                with self.cam_lock:
                    ret, frame = self.cap.read()
                if not ret:
                    print("CaptureThread: Unable to read frame")
                    time.sleep(0.5)
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                focusVal = advancedFocusMetric(gray)
                focusValScaled = np.clip(focusVal, 0, 100)
                self.captureLog[self.currentCaptureIndex, :] = [commandedPos, actualPos, errorVal, adjustment, focusValScaled]
                self.rotorAngle += 5 * math.pi / 180.0
                self.queue.put(('update', commandedPos, actualPos, errorVal, adjustment, focusValScaled, gray, self.rotorAngle))
                idx = int(np.argmin(np.abs(self.stagePositions - commandedPosNext)))
                self.currentCaptureIndex = idx
                time.sleep(0.5)
            self.queue.put(('done',))
        except Exception as e:
            print("Exception in CaptureThread:")
            traceback.print_exc()
            self.queue.put(('done',))

# ---------------------------
# Main Simulation GUI using PyQt5
# ---------------------------
class MotorCameraSimulation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motor-Camera System Simulation")
        self.setGeometry(100, 100, 1000, 600)
        
        # Simulation parameters
        self.stagePositions = np.arange(0, 1001, 10)  # 0 to 1000 microns
        self.numPos = len(self.stagePositions)
        self.focusThreshold = 0.5
        
        self.UB = np.nan
        self.LB = np.nan
        self.logScan = None  # to be filled after scanning
        
        # Create a global camera capture object and a lock
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Unable to open camera")
            sys.exit(1)
        self.cam_lock = threading.Lock()
        
        # Setup fuzzy inference systems (control systems)
        self.setupFocusFIS()
        self.setupMotorFIS()
        
        # Setup GUI
        self.initUI()
        
        # Queues for inter-thread communication
        self.scanQueue = None
        self.captureQueue = None
        
        # Timer to poll queues from worker threads
        self.pollTimer = QtCore.QTimer()
        self.pollTimer.timeout.connect(self.pollQueues)
        self.pollTimer.start(50)  # poll every 50 ms
        
        # Worker threads will be started via threading.Thread
        self.scanThread = None
        self.captureThread = None
    
    def initUI(self):
        widget = QWidget()
        self.setCentralWidget(widget)
        
        # Camera image display using Matplotlib
        self.figImage, self.axImage = plt.subplots(figsize=(4, 3))
        self.axImage.axis('off')
        self.canvasImage = FigureCanvas(self.figImage)
        
        # Stage view plot
        self.figStage, self.axStage = plt.subplots(figsize=(1.5, 3))
        self.axStage.set_title("Linear Stage Position")
        self.axStage.set_xlabel("Stage")
        self.axStage.set_ylabel("Position (microns)")
        self.axStage.set_xlim(0, 1)
        self.axStage.set_ylim(0, 1000)
        self.axStage.plot([0.5, 0.5], [0, 1000], 'k-', linewidth=2)
        self.stageMarker, = self.axStage.plot(0.5, 0, 'ro', markersize=8)
        
        # Rotor animation plot
        self.figRotor, self.axRotor = plt.subplots(figsize=(1, 1))
        self.axRotor.set_title("Motor Rotor")
        self.axRotor.set_xlim(-1.2, 1.2)
        self.axRotor.set_ylim(-1.2, 1.2)
        self.axRotor.set_aspect('equal')
        theta = np.linspace(0, 2*math.pi, 100)
        self.axRotor.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
        self.rotorQuiver = self.axRotor.quiver(0, 0, 0.8, 0, color='r', angles='xy', scale_units='xy', scale=1)
        
        # Labels and buttons
        self.lblUB = QLabel("Upper Bound (UB): TBD")
        self.lblLB = QLabel("Lower Bound (LB): TBD")
        self.lblEncoder = QLabel("Encoder: 0.0 microns")
        self.btnScan = QPushButton("Start Scan")
        self.btnScan.clicked.connect(self.startScan)
        self.btnCapture = QPushButton("Start Capture")
        self.btnCapture.setEnabled(False)
        self.btnCapture.clicked.connect(self.startCapture)
        
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvasImage)
        leftLayout.addWidget(self.btnScan)
        leftLayout.addWidget(self.btnCapture)
        leftLayout.addWidget(self.lblEncoder)
        leftLayout.addWidget(self.lblUB)
        leftLayout.addWidget(self.lblLB)
        
        stageCanvas = FigureCanvas(self.figStage)
        rotorCanvas = FigureCanvas(self.figRotor)
        rightLayout = QVBoxLayout()
        rightLayout.addWidget(stageCanvas)
        rightLayout.addWidget(rotorCanvas)
        
        mainLayout = QHBoxLayout()
        mainLayout.addLayout(leftLayout)
        mainLayout.addLayout(rightLayout)
        widget.setLayout(mainLayout)
    
    def setupFocusFIS(self):
        # Create FocusFIS control system using scikit-fuzzy
        self.focus_metric = ctrl.Antecedent(np.arange(0, 101, 1), 'FocusMetric')
        self.focus_degree = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'FocusDegree')
        self.focus_metric['Blurred'] = fuzz.trimf(self.focus_metric.universe, [0, 0, 40])
        self.focus_metric['Moderate'] = fuzz.trimf(self.focus_metric.universe, [30, 50, 70])
        self.focus_metric['Sharp'] = fuzz.trimf(self.focus_metric.universe, [60, 100, 100])
        self.focus_degree['Low'] = fuzz.trimf(self.focus_degree.universe, [0, 0, 0.3])
        self.focus_degree['Medium'] = fuzz.trimf(self.focus_degree.universe, [0.2, 0.5, 0.8])
        self.focus_degree['High'] = fuzz.trimf(self.focus_degree.universe, [0.7, 1, 1])
        rule1 = ctrl.Rule(self.focus_metric['Blurred'], self.focus_degree['Low'])
        rule2 = ctrl.Rule(self.focus_metric['Moderate'], self.focus_degree['Medium'])
        rule3 = ctrl.Rule(self.focus_metric['Sharp'], self.focus_degree['High'])
        self.focus_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    
    def setupMotorFIS(self):
        # Create MotorFIS control system using scikit-fuzzy
        self.error = ctrl.Antecedent(np.arange(-2, 2.01, 0.01), 'Error')
        self.adjustment = ctrl.Consequent(np.arange(-2, 2.01, 0.01), 'Adjustment')
        self.error['Negative'] = fuzz.trimf(self.error.universe, [-2, -2, 0])
        self.error['Zero'] = fuzz.trimf(self.error.universe, [-1, 0, 1])
        self.error['Positive'] = fuzz.trimf(self.error.universe, [0, 2, 2])
        self.adjustment['Increase'] = fuzz.trimf(self.adjustment.universe, [-2, -2, 0])
        self.adjustment['NoChange'] = fuzz.trimf(self.adjustment.universe, [-0.5, 0, 0.5])
        self.adjustment['Decrease'] = fuzz.trimf(self.adjustment.universe, [0, 2, 2])
        rule1 = ctrl.Rule(self.error['Negative'], self.adjustment['Increase'])
        rule2 = ctrl.Rule(self.error['Zero'], self.adjustment['NoChange'])
        rule3 = ctrl.Rule(self.error['Positive'], self.adjustment['Decrease'])
        self.motor_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    
    def pollQueues(self):
        # Poll scan queue if active
        if self.scanQueue is not None:
            try:
                while True:
                    item = self.scanQueue.get_nowait()
                    if item[0] == 'update':
                        _, z, focusValScaled, fuzzyOutput, gray = item
                        self.update_scan(z, focusValScaled, fuzzyOutput, gray)
                    elif item[0] == 'done':
                        self.scanQueue = None
                        self.scan_finished()
                        break
            except queue.Empty:
                pass
        # Poll capture queue if active
        if self.captureQueue is not None:
            try:
                while True:
                    item = self.captureQueue.get_nowait()
                    if item[0] == 'update':
                        (_, commandedPos, actualPos, errorVal, adjustment,
                         focusValScaled, gray, rotorAngle) = item
                        self.update_capture(commandedPos, actualPos, errorVal, adjustment, focusValScaled, gray, rotorAngle)
                    elif item[0] == 'done':
                        self.captureQueue = None
                        self.capture_finished()
                        break
            except queue.Empty:
                pass
    
    # GUI update functions
    def update_scan(self, z, focusValScaled, fuzzyOutput, gray):
        self.axImage.clear()
        self.axImage.imshow(gray, cmap='gray')
        self.axImage.set_title(f"Scan: Z = {z} microns, Focus = {focusValScaled:.1f}")
        self.axImage.axis('off')
        self.canvasImage.draw()
        self.lblEncoder.setText(f"Encoder: {z} microns")
        self.stageMarker.set_data(0.5, z)
        self.axStage.plot(z, focusValScaled, 'b*')
        self.figStage.canvas.draw()
    
    def scan_finished(self):
        fuzzyOut = self.scanThread.logScan[:, 2]
        idxInFocus = np.where(fuzzyOut > self.focusThreshold)[0]
        if idxInFocus.size > 0:
            self.UB = self.stagePositions[idxInFocus[0]]
            self.LB = self.stagePositions[idxInFocus[-1]]
        self.lblUB.setText(f"Upper Bound (UB): {self.UB:.1f} microns")
        self.lblLB.setText(f"Lower Bound (LB): {self.LB:.1f} microns")
        np.savetxt(os.path.join(os.getcwd(), "scanLog.csv"), self.scanThread.logScan, delimiter=",")
        self.logScan = self.scanThread.logScan
        self.btnCapture.setEnabled(True)
    
    def update_capture(self, commandedPos, actualPos, errorVal, adjustment, focusValScaled, gray, rotorAngle):
        self.axImage.clear()
        self.axImage.imshow(gray, cmap='gray')
        self.axImage.set_title(f"Capture: Z = {actualPos:.1f} microns, Focus = {focusValScaled:.1f}")
        self.axImage.axis('off')
        self.canvasImage.draw()
        self.lblEncoder.setText(f"Encoder: {actualPos:.1f} microns (Error: {errorVal:.2f}, Adj: {adjustment:.2f})")
        self.stageMarker.set_data(0.5, actualPos)
        U = 0.8 * math.cos(rotorAngle)
        V = 0.8 * math.sin(rotorAngle)
        self.rotorQuiver.set_UVC(U, V)
        self.figRotor.canvas.draw()
    
    def capture_finished(self):
        np.savetxt(os.path.join(os.getcwd(), "captureLog.csv"), self.captureThread.captureLog, delimiter=",")
        QMessageBox.information(self, "Done", "Capture phase complete!")
        self.displayAndSaveFIS()
    
    def startScan(self):
        self.btnScan.setEnabled(False)
        self.scanThread = ScanThread(self.stagePositions, self.focusThreshold, self.focus_ctrl, self.cap, self.cam_lock)
        self.scanQueue = self.scanThread.queue
        threading.Thread(target=self.scanThread.run, daemon=True).start()
    
    def startCapture(self):
        self.btnCapture.setEnabled(False)
        self.captureThread = CaptureThread(self.stagePositions, self.UB, self.LB, self.motor_ctrl, self.logScan, self.cap, self.cam_lock)
        self.captureQueue = self.captureThread.queue
        threading.Thread(target=self.captureThread.run, daemon=True).start()
    
    def displayAndSaveFIS(self):
        visFolder = os.path.join(os.getcwd(), "visualizations")
        if not os.path.exists(visFolder):
            os.makedirs(visFolder)
        # FocusFIS plots
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,8))
        x_focus = np.arange(0, 101, 1)
        ax1.plot(x_focus, fuzz.trimf(x_focus, [0, 0, 40]), label='Blurred')
        ax1.plot(x_focus, fuzz.trimf(x_focus, [30, 50, 70]), label='Moderate')
        ax1.plot(x_focus, fuzz.trimf(x_focus, [60, 100, 100]), label='Sharp')
        ax1.set_title('FocusFIS - Input: FocusMetric')
        ax1.legend()
        x_degree = np.arange(0, 1.01, 0.01)
        ax2.plot(x_degree, fuzz.trimf(x_degree, [0, 0, 0.3]), label='Low')
        ax2.plot(x_degree, fuzz.trimf(x_degree, [0.2, 0.5, 0.8]), label='Medium')
        ax2.plot(x_degree, fuzz.trimf(x_degree, [0.7, 1, 1]), label='High')
        ax2.set_title('FocusFIS - Output: FocusDegree')
        ax2.legend()
        fig1.savefig(os.path.join(visFolder, "FocusFIS_MF.png"))
        plt.close(fig1)
        
        # MotorFIS plots
        fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(6,8))
        x_error = np.arange(-2, 2.01, 0.01)
        ax3.plot(x_error, fuzz.trimf(x_error, [-2, -2, 0]), label='Negative')
        ax3.plot(x_error, fuzz.trimf(x_error, [-1, 0, 1]), label='Zero')
        ax3.plot(x_error, fuzz.trimf(x_error, [0, 2, 2]), label='Positive')
        ax3.set_title('MotorFIS - Input: Error')
        ax3.legend()
        x_adjust = np.arange(-2, 2.01, 0.01)
        ax4.plot(x_adjust, fuzz.trimf(x_adjust, [-2, -2, 0]), label='Increase')
        ax4.plot(x_adjust, fuzz.trimf(x_adjust, [-0.5, 0, 0.5]), label='NoChange')
        ax4.plot(x_adjust, fuzz.trimf(x_adjust, [0, 2, 2]), label='Decrease')
        ax4.set_title('MotorFIS - Output: Adjustment')
        ax4.legend()
        fig2.savefig(os.path.join(visFolder, "MotorFIS_MF.png"))
        plt.close(fig2)

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    sim = MotorCameraSimulation()
    sim.show()
    sys.exit(app.exec_())
