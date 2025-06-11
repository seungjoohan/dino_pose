// Pose app entirely written by Cursor
import SwiftUI
import CoreML
import Vision
import AVFoundation

@main
struct PoseTestApp: App {
    var body: some Scene {
        WindowGroup {
            ModelSelectionView()
        }
    }
}

// MARK: - Models and Data Structures
struct MLModelInfo: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let path: String
    let family: String
}

struct Keypoint {
    let x: Double
    let y: Double
    let confidence: Double
}

// MARK: - Model Selection View
struct ModelSelectionView: View {
    @State private var availableModels: [MLModelInfo] = []
    @State private var selectedModel: MLModelInfo?
    @State private var isScanning = false
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                Text("🎯 Pose Estimation Models")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .padding()
                
                Text("Select a model to start pose estimation")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                
                if isScanning {
                    ProgressView("Scanning for models...")
                        .padding()
                } else if availableModels.isEmpty {
                    ContentUnavailableView(
                        "No Models Found",
                        systemImage: "doc.questionmark",
                        description: Text("No .mlpackage files found in test_models directory")
                    )
                    .padding()
                } else {
                    List(availableModels, id: \.self) { model in
                        ModelRowView(model: model, isSelected: selectedModel?.id == model.id) {
                            selectedModel = model
                        }
                    }
                    .listStyle(.plain)
                }
                
                if let selectedModel = selectedModel {
                    VStack {
                        Text("Selected: \(selectedModel.name)")
                            .font(.headline)
                        
                        Text("Family: \(selectedModel.family)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        
                        NavigationLink(destination: RealTimePoseView(modelInfo: selectedModel)) {
                            Label("Start Posing!", systemImage: "camera.fill")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(.blue)
                                .foregroundStyle(.white)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                        }
                        .padding(.horizontal)
                    }
                    .padding()
                }
                
                Spacer()
            }
            .task {
                await scanForModels()
            }
        }
    }
    
    private func scanForModels() async {
        isScanning = true
        
        let models = await Task.detached {
            discoverModels()
        }.value
        
        await MainActor.run {
            self.availableModels = models
            self.isScanning = false
        }
    }
    
    private func discoverModels() -> [MLModelInfo] {
        var models: [MLModelInfo] = []
        
        guard let bundlePath = Bundle.main.path(forResource: "test_models", ofType: nil) else {
            print("❌ test_models directory not found in bundle")
            print("📁 Bundle resources:")
            if let bundleResources = Bundle.main.urls(forResourcesWithExtension: nil, subdirectory: nil) {
                for resource in bundleResources.prefix(10) {
                    print("   - \(resource.lastPathComponent)")
                }
            }
            return models
        }
        
        let fileManager = FileManager.default
        
        do {
            let subdirectories = try fileManager.contentsOfDirectory(atPath: bundlePath)
            
            for subdirectory in subdirectories {
                let subdirPath = "\(bundlePath)/\(subdirectory)"
                var isDirectory: ObjCBool = false
                
                if fileManager.fileExists(atPath: subdirPath, isDirectory: &isDirectory),
                   isDirectory.boolValue {
                    
                    let files = try fileManager.contentsOfDirectory(atPath: subdirPath)
                    
                    for file in files {
                        if file.hasSuffix(".mlpackage") {
                            let modelName = String(file.dropLast(10))
                            let family = detectModelFamily(from: modelName)
                            
                            let model = MLModelInfo(
                                name: modelName,
                                path: "\(subdirPath)/\(file)",
                                family: family
                            )
                            models.append(model)
                        }
                    }
                }
            }
        } catch {
            print("❌ Error scanning models: \(error)")
        }
        
        return models.sorted { $0.name < $1.name }
    }
    
    private func detectModelFamily(from name: String) -> String {
        let lowercased = name.lowercased()
        if lowercased.contains("dino") {
            return "DINO"
        } else if lowercased.contains("fastvit") {
            return "FastViT"
        } else {
            return "Unknown"
        }
    }
}

// MARK: - Model Row View
struct ModelRowView: View {
    let model: MLModelInfo
    let isSelected: Bool
    let onTap: () -> Void
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(model.name)
                    .font(.headline)
                
                Text(model.family)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            
            Spacer()
            
            if isSelected {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.blue)
            }
        }
        .padding(.vertical, 8)
        .contentShape(Rectangle())
        .onTapGesture {
            onTap()
        }
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isSelected ? .blue.opacity(0.1) : .clear)
        )
    }
}

// MARK: - Real Time Pose View
struct RealTimePoseView: View {
    let modelInfo: MLModelInfo
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var poseEstimator: PoseEstimator
    @State private var currentKeypoints: [Keypoint] = []
    @State private var inferenceTime: Double = 0
    @State private var avgConfidence: Double = 0
    @State private var lastProcessTime: CFAbsoluteTime = 0
    @State private var actualFPS: Double = 0
    @State private var lastFPSUpdate: CFAbsoluteTime = 0
    @State private var frameCount: Int = 0
    @Environment(\.dismiss) private var dismiss
    
    // Frame processing counter
    private static var frameLogCounter = 0
    
    init(modelInfo: MLModelInfo) {
        self.modelInfo = modelInfo
        self._poseEstimator = StateObject(wrappedValue: PoseEstimator(modelPath: modelInfo.path))
    }
    
    var body: some View {
        ZStack {
            // Camera Preview
            CameraPreviewView(session: cameraManager.captureSession)
                .ignoresSafeArea()
            
            // Skeleton Overlay
            if poseEstimator.isModelReady && avgConfidence >= 0.5 {
                SkeletonOverlayView(keypoints: currentKeypoints)
            }
            
            // Info Overlay
            VStack {
                HStack {
                    Button(action: { dismiss() }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title)
                            .foregroundStyle(.white)
                            .background(.black.opacity(0.6), in: Circle())
                    }
                    
                    Spacer()
                    
                    VStack(alignment: .trailing) {
                        InfoChip(text: modelInfo.name)
                        
                        InfoChip(text: "Input: \(Int(poseEstimator.getInputSize().width))x\(Int(poseEstimator.getInputSize().height))")
                        
                        InfoChip(
                            text: poseEstimator.isModelReady ? "Model Ready" : "Loading...",
                            color: poseEstimator.isModelReady ? .green : .orange
                        )
                        
                        InfoChip(text: "FPS: \(String(format: "%.1f", actualFPS))")
                        
                        InfoChip(text: "Theory: \(String(format: "%.0f", 1000.0 / max(inferenceTime, 1)))", color: .gray)
                        
                        InfoChip(
                            text: "Conf: \(String(format: "%.2f", avgConfidence))",
                            color: avgConfidence >= 0.5 ? .green : .orange
                        )
                        
                        InfoChip(
                            text: cameraManager.currentCameraPosition == .front ? "Front" : "Back",
                            color: .blue
                        )
                    }
                }
                .padding()
                
                Spacer()
                
                // Camera Switch Button
                HStack {
                    Button(action: {
                        Task {
                            await cameraManager.switchCamera()
                        }
                    }) {
                        Image(systemName: "camera.rotate.fill")
                            .font(.title2)
                            .foregroundStyle(.white)
                            .frame(width: 50, height: 50)
                            .background(.black.opacity(0.6), in: Circle())
                    }
                    
                    Spacer()
                }
                .padding(.horizontal)
                
                Spacer()
                
                if !poseEstimator.isModelReady {
                    Text("🤖 Loading AI model...")
                        .font(.headline)
                        .foregroundStyle(.white)
                        .padding()
                        .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 12))
                        .padding()
                } else if avgConfidence < 0.5 {
                    Text("📍 Move to get better pose detection")
                        .font(.headline)
                        .foregroundStyle(.white)
                        .padding()
                        .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 12))
                        .padding()
                }
            }
        }
        .navigationBarHidden(true)
        .task {
            await cameraManager.startSession()
        }
        .onDisappear {
            Task {
                await cameraManager.stopSession()
            }
        }
        .onReceive(cameraManager.$latestFrame) { pixelBuffer in
            if let buffer = pixelBuffer, poseEstimator.isModelReady {
                processFrame(buffer, with: poseEstimator)
            } else {
                if pixelBuffer == nil {
                    print("⚠️ No pixel buffer received from camera")
                }
                if !poseEstimator.isModelReady {
                    print("⚠️ Model not ready yet")
                }
            }
        }
    }
    
    private func processFrame(_ pixelBuffer: CVPixelBuffer, with estimator: PoseEstimator) {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Add throttling to avoid overwhelming the system
        let currentTime = CFAbsoluteTimeGetCurrent()
        
        // Measure actual maximum performance (no throttling)
        // if currentTime - lastProcessTime < 0.033 {
        //     return
        // }
        lastProcessTime = currentTime
        
        // Reduce logging overhead
        // RealTimePoseView.frameLogCounter += 1
        // if RealTimePoseView.frameLogCounter % 60 == 0 {
        //     print("🎥 Processing frame... (\(RealTimePoseView.frameLogCounter) frames processed )")
        // }
        
        estimator.estimatePose(from: pixelBuffer) { keypoints, confidence in
            let endTime = CFAbsoluteTimeGetCurrent()
            let timeElapsed = (endTime - startTime) * 1000
            
                    Task { @MainActor in
            self.currentKeypoints = keypoints
            self.avgConfidence = confidence
            self.inferenceTime = timeElapsed
            
            // Calculate processing FPS (inference completed frames)
            self.frameCount += 1
            let timeSinceLastUpdate = endTime - self.lastFPSUpdate
            if timeSinceLastUpdate >= 1.0 { // Update every second
                self.actualFPS = Double(self.frameCount) / timeSinceLastUpdate
                self.lastFPSUpdate = endTime
                self.frameCount = 0
                print("🚀 PROCESSING FPS: \(String(format: "%.1f", self.actualFPS)) (completed inference)")
            }
        }
        }
    }
}

// MARK: - Info Chip View
struct InfoChip: View {
    let text: String
    let color: Color
    
    init(text: String, color: Color = .white) {
        self.text = text
        self.color = color
    }
    
    var body: some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(color)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(.black.opacity(0.6), in: Capsule())
    }
}

// MARK: - Camera Preview View (iOS 17+ Modern SwiftUI)
struct CameraPreviewView: View {
    let session: AVCaptureSession?
    
    var body: some View {
        if let session = session {
            CameraPreviewRepresentable(session: session)
        } else {
            Rectangle()
                .fill(.black)
                .overlay {
                    Text("Camera Loading...")
                        .foregroundStyle(.white)
                }
        }
    }
}

// Using UIViewRepresentable without explicit UIKit import
struct CameraPreviewRepresentable: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> CameraPreviewUIView {
        let view = CameraPreviewUIView()
        view.session = session
        return view
    }
    
    func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {
        uiView.session = session
    }
}

class CameraPreviewUIView: UIView {
    var session: AVCaptureSession? {
        didSet {
            setupPreviewLayer()
        }
    }
    
    private var previewLayer: AVCaptureVideoPreviewLayer?
    
    override func didMoveToSuperview() {
        super.didMoveToSuperview()
        setupPreviewLayer()
    }
    
    private func setupPreviewLayer() {
        guard let session = session else { return }
        
        previewLayer?.removeFromSuperlayer()
        
        let newPreviewLayer = AVCaptureVideoPreviewLayer(session: session)
        newPreviewLayer.videoGravity = .resizeAspectFill
        newPreviewLayer.frame = bounds
        
        layer.addSublayer(newPreviewLayer)
        self.previewLayer = newPreviewLayer
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer?.frame = bounds
    }
}

// MARK: - Camera Manager
class CameraManager: NSObject, ObservableObject {
    @Published var latestFrame: CVPixelBuffer?
    @Published var currentCameraPosition: AVCaptureDevice.Position = .front
    var captureSession: AVCaptureSession?
    
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private var currentInput: AVCaptureDeviceInput?
    
    func startSession() async {
        await withCheckedContinuation { continuation in
            sessionQueue.async {
                print("🎬 Starting camera session setup...")
                self.setupCaptureSession()
                print("🎬 Camera session setup completed")
                continuation.resume()
            }
        }
    }
    
    func stopSession() async {
        await withCheckedContinuation { continuation in
            sessionQueue.async {
                self.captureSession?.stopRunning()
                continuation.resume()
            }
        }
    }
    
    func switchCamera() async {
        await withCheckedContinuation { continuation in
            sessionQueue.async {
                self.switchCameraSync()
                continuation.resume()
            }
        }
    }
    
    private func setupCaptureSession() {
        // ✅ Check camera permission first
        let authStatus = AVCaptureDevice.authorizationStatus(for: .video)
        print("📷 Camera permission status: \(authStatus.rawValue)")
        
        switch authStatus {
        case .authorized:
            print("✅ Camera permission granted")
        case .notDetermined:
            print("⚠️ Camera permission not determined, requesting...")
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    print("✅ Camera permission granted after request")
                    DispatchQueue.global(qos: .userInitiated).async {
                        self.setupCaptureSessionInternal()
                    }
                } else {
                    print("❌ Camera permission denied")
                }
            }
            return
        case .denied, .restricted:
            print("❌ Camera permission denied or restricted")
            return
        @unknown default:
            print("❌ Unknown camera permission status")
            return
        }
        
        setupCaptureSessionInternal()
    }
    
    private func setupCaptureSessionInternal() {
        let session = AVCaptureSession()
        
        // ✅ CRITICAL: Avoid session presets that can limit frame rate
        // Instead use AVCaptureSessionPresetInputPriority to let device format control
        session.sessionPreset = .inputPriority
        print("🎥 Using session preset: .inputPriority (device format controls quality)")
        
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: currentCameraPosition),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            print("❌ Failed to create camera or input")
            return
        }
        
        // ✅ IMPORTANT: Begin configuration before making changes
        session.beginConfiguration()
        
        // Configure camera for higher frame rate with better targeting
        do {
            try camera.lockForConfiguration()
            
            print("📋 Available formats: \(camera.formats.count)")
            print("📱 Device: \(camera.localizedName)")
            
            // ✅ NEW: Find optimal format for 60 FPS specifically
            var bestFormat: AVCaptureDevice.Format?
            var targetFrameRate: Double = 60  // Target 60 FPS primarily
            
            // Sort formats by preference: higher FPS first, then reasonable resolution
            let sortedFormats = camera.formats.sorted { (format1: AVCaptureDevice.Format, format2: AVCaptureDevice.Format) in
                let dim1 = CMVideoFormatDescriptionGetDimensions(format1.formatDescription)
                let dim2 = CMVideoFormatDescriptionGetDimensions(format2.formatDescription)
                let fps1 = format1.videoSupportedFrameRateRanges.first?.maxFrameRate ?? 0
                let fps2 = format2.videoSupportedFrameRateRanges.first?.maxFrameRate ?? 0
                
                // Prefer higher FPS, then consider resolution
                if fps1 != fps2 {
                    return fps1 > fps2
                }
                // If FPS is same, prefer reasonable resolution (720p-1080p range)
                let area1 = Int(dim1.width) * Int(dim1.height)
                let area2 = Int(dim2.width) * Int(dim2.height)
                let ideal = 1280 * 720  // 720p as sweet spot
                return abs(area1 - ideal) < abs(area2 - ideal)
            }
            
            // Find best format with detailed analysis
            for (index, format) in sortedFormats.enumerated() {
                let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                let width = dimensions.width
                let height = dimensions.height
                let pixelFormat = CMFormatDescriptionGetMediaSubType(format.formatDescription)
                
                for range in format.videoSupportedFrameRateRanges {
                    let maxFPS = range.maxFrameRate
                    let minFPS = range.minFrameRate
                    
                    print("🎯 Format #\(index): \(width)x\(height) @ \(minFPS)-\(maxFPS) FPS (format: \(String(format: "%c%c%c%c", (pixelFormat >> 24) & 0xff, (pixelFormat >> 16) & 0xff, (pixelFormat >> 8) & 0xff, pixelFormat & 0xff)))")
                    
                    // Target criteria for optimal performance:
                    // 1. Support 60 FPS or close to it
                    // 2. Resolution between 480p-1080p (not too low, not too high)
                    // 3. Prefer 420v/420f formats for better performance
                    if maxFPS >= 50 && width >= 640 && width <= 1920 && height >= 480 && height <= 1080 {
                        if bestFormat == nil || maxFPS > targetFrameRate {
                            bestFormat = format
                            targetFrameRate = maxFPS
                            print("✨ NEW BEST: \(width)x\(height) @ \(maxFPS) FPS")
                        }
                    }
                }
            }
            
            // ✅ Apply the optimal format
            if let format = bestFormat {
                camera.activeFormat = format
                
                // ✅ CRITICAL: Set frame rate precisely to avoid 24 FPS trap
                let desiredFPS = min(targetFrameRate, 60)  // Cap at 60 FPS
                let frameDuration = CMTimeMake(value: 1, timescale: Int32(desiredFPS))
                
                // Verify the format actually supports this frame rate
                let supportedRanges = format.videoSupportedFrameRateRanges
                var canSetDesiredFPS = false
                for range in supportedRanges {
                    if desiredFPS >= range.minFrameRate && desiredFPS <= range.maxFrameRate {
                        canSetDesiredFPS = true
                        break
                    }
                }
                
                if canSetDesiredFPS {
                    camera.activeVideoMinFrameDuration = frameDuration
                    camera.activeVideoMaxFrameDuration = frameDuration
                    print("✅ Camera configured for \(desiredFPS) FPS with format: \(bestFormat!)")
                } else {
                    print("⚠️ Desired FPS \(desiredFPS) not supported by format, using format default")
                }
                
                // ✅ Additional optimizations to prevent 24 FPS limitation
                // Disable auto exposure duration limits that can cause 24 FPS cap
                if camera.isExposureModeSupported(.custom) {
                    // Don't use custom exposure as it can interfere, but ensure auto exposure is set
                    camera.exposureMode = .continuousAutoExposure
                }
                
                // ✅ Disable video stabilization which can force 24 FPS in some cases
                // (based on search results about 1/40s exposure limits)
                
            } else {
                print("❌ No suitable high frame rate format found!")
                // Fallback: try to find any format that supports >30 FPS
                for format in camera.formats {
                    for range in format.videoSupportedFrameRateRanges {
                        if range.maxFrameRate > 30 {
                            camera.activeFormat = format
                            camera.activeVideoMinFrameDuration = CMTimeMake(value: 1, timescale: Int32(range.maxFrameRate))
                            camera.activeVideoMaxFrameDuration = CMTimeMake(value: 1, timescale: Int32(range.maxFrameRate))
                            print("📦 Fallback: Using \(format) at \(range.maxFrameRate) FPS")
                            break
                        }
                    }
                    if bestFormat != nil { break }
                }
            }
            
            camera.unlockForConfiguration()
        } catch {
            print("❌ Failed to configure camera: \(error)")
        }
        
        // ✅ Add input to session
        if session.canAddInput(input) {
            session.addInput(input)
            currentInput = input
            print("✅ Camera input added successfully")
        } else {
            print("❌ Cannot add camera input to session")
            session.commitConfiguration()
            return
        }
        
        // Configure video output for optimal performance
        videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        
        // Optimize for real-time processing
        videoOutput.alwaysDiscardsLateVideoFrames = true
        
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
            print("✅ Video output added successfully")
            
            // ✅ IMPORTANT: Configure connection to prevent stabilization-induced FPS limits
            if let connection = videoOutput.connection(with: .video) {
                // Disable video stabilization that can cause 24 FPS limitation
                if connection.isVideoStabilizationSupported {
                    connection.preferredVideoStabilizationMode = .off
                    print("📱 Video stabilization: DISABLED (prevents FPS limitations)")
                }
            }
        } else {
            print("❌ Cannot add video output")
            session.commitConfiguration()
            return
        }
        
        updateCameraOrientation()
        
        // ✅ Commit configuration before setting session
        session.commitConfiguration()
        print("🔧 Session configuration committed")
        
        // ✅ Set session BEFORE starting
        DispatchQueue.main.async {
            self.captureSession = session
            print("📱 Capture session set on main thread")
        }
        
        // ✅ CRITICAL: Actually start the session
        print("🚀 Starting capture session...")
        session.startRunning()
        print("✅ Capture session started: \(session.isRunning)")
    }
    
    private func switchCameraSync() {
        guard let session = captureSession else { return }
        
        session.beginConfiguration()
        
        // Remove current input
        if let currentInput = currentInput {
            session.removeInput(currentInput)
        }
        
        // Switch camera position
        let newPosition: AVCaptureDevice.Position = (currentCameraPosition == .front) ? .back : .front
        
        guard let newCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: newPosition),
              let newInput = try? AVCaptureDeviceInput(device: newCamera) else {
            session.commitConfiguration()
            return
        }
        
        // ✅ Apply same FPS optimization for switched camera
        do {
            try newCamera.lockForConfiguration()
            
            // Find best format for high FPS (same logic as setupCaptureSession)
            var bestFormat: AVCaptureDevice.Format?
            var targetFrameRate: Double = 60
            
            let sortedFormats = newCamera.formats.sorted { (format1: AVCaptureDevice.Format, format2: AVCaptureDevice.Format) in
                let fps1 = format1.videoSupportedFrameRateRanges.first?.maxFrameRate ?? 0
                let fps2 = format2.videoSupportedFrameRateRanges.first?.maxFrameRate ?? 0
                return fps1 > fps2
            }
            
            for format in sortedFormats {
                let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                let width = dimensions.width
                let height = dimensions.height
                
                for range in format.videoSupportedFrameRateRanges {
                    if range.maxFrameRate >= 50 && width >= 640 && width <= 1920 && height >= 480 && height <= 1080 {
                        if bestFormat == nil || range.maxFrameRate > targetFrameRate {
                            bestFormat = format
                            targetFrameRate = range.maxFrameRate
                        }
                    }
                }
            }
            
            if let format = bestFormat {
                newCamera.activeFormat = format
                let desiredFPS = min(targetFrameRate, 60)
                let frameDuration = CMTimeMake(value: 1, timescale: Int32(desiredFPS))
                newCamera.activeVideoMinFrameDuration = frameDuration
                newCamera.activeVideoMaxFrameDuration = frameDuration
                print("✅ Switched camera configured for \(desiredFPS) FPS")
            }
            
            // Set continuous auto exposure
            if newCamera.isExposureModeSupported(.continuousAutoExposure) {
                newCamera.exposureMode = .continuousAutoExposure
            }
            
            newCamera.unlockForConfiguration()
        } catch {
            print("❌ Failed to configure switched camera: \(error)")
        }
        
        if session.canAddInput(newInput) {
            session.addInput(newInput)
            currentInput = newInput
            
            DispatchQueue.main.async {
                self.currentCameraPosition = newPosition
            }
        }
        
        updateCameraOrientation()
        session.commitConfiguration()
    }
    
    private func updateCameraOrientation() {
        if let connection = videoOutput.connection(with: .video) {
            if #available(iOS 17.0, *) {
                connection.videoRotationAngle = 90.0
            } else {
                connection.videoOrientation = .portrait
            }
            // Only mirror front camera
            connection.isVideoMirrored = (currentCameraPosition == .front)
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    private static var cameraFrameCount = 0
    private static var lastCameraFPSUpdate: CFAbsoluteTime = 0
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Track actual camera frame rate
        CameraManager.cameraFrameCount += 1
        let currentTime = CFAbsoluteTimeGetCurrent()
        
        if CameraManager.lastCameraFPSUpdate == 0 {
            CameraManager.lastCameraFPSUpdate = currentTime
        }
        
        let timeSinceLastUpdate = currentTime - CameraManager.lastCameraFPSUpdate
        if timeSinceLastUpdate >= 1.0 {
            let cameraFPS = Double(CameraManager.cameraFrameCount) / timeSinceLastUpdate
            print("📷 CAMERA FPS: \(String(format: "%.1f", cameraFPS))")
            CameraManager.cameraFrameCount = 0
            CameraManager.lastCameraFPSUpdate = currentTime
        }
        
        DispatchQueue.main.async {
            self.latestFrame = pixelBuffer
        }
    }
}

// MARK: - Skeleton Overlay View
struct SkeletonOverlayView: View {
    let keypoints: [Keypoint]
    
    // Project-specific keypoint connections
    private let connections: [(Int, Int)] = [
        (0, 1), (1, 24), (7, 6), (6, 5), (5, 24), (24, 2), (2, 3), (3, 4),
        (24, 19), (19, 25), (25, 11), (25, 8), (11, 12), (12, 13), (8, 9), (9, 10),
        (14, 15), (15, 16), (14, 17), (17, 18), (4, 20), (10, 21), (7, 22), (13, 23)
    ]
    
    var body: some View {
        Canvas { context, size in
            let virtualKeypoints = calculateVirtualKeypoints()
            
            // Draw skeleton connections
            for (startIdx, endIdx) in connections {
                let startKeypoint = getKeypoint(at: startIdx, virtual: virtualKeypoints)
                let endKeypoint = getKeypoint(at: endIdx, virtual: virtualKeypoints)
                
                if startKeypoint.confidence > 0.3 && endKeypoint.confidence > 0.3 {
                    let startPoint = CGPoint(
                        x: startKeypoint.x * size.width,
                        y: startKeypoint.y * size.height
                    )
                    let endPoint = CGPoint(
                        x: endKeypoint.x * size.width,
                        y: endKeypoint.y * size.height
                    )
                    
                    var path = Path()
                    path.move(to: startPoint)
                    path.addLine(to: endPoint)
                    
                    context.stroke(path, with: .color(.cyan), style: StrokeStyle(lineWidth: 3, lineCap: .round))
                }
            }
            
            // Draw keypoints
            for keypoint in keypoints {
                let point = CGPoint(x: keypoint.x * size.width, y: keypoint.y * size.height)
                let color: Color = keypoint.confidence > 0.7 ? .green : .yellow
                let radius: CGFloat = keypoint.confidence > 0.7 ? 8 : 6
                
                context.fill(
                    Path(ellipseIn: CGRect(x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2)),
                    with: .color(color)
                )
            }
            
            // Draw virtual keypoints
            for virtualKeypoint in virtualKeypoints.values {
                if virtualKeypoint.confidence > 0.3 {
                    let point = CGPoint(x: virtualKeypoint.x * size.width, y: virtualKeypoint.y * size.height)
                    context.fill(
                        Path(ellipseIn: CGRect(x: point.x - 4, y: point.y - 4, width: 8, height: 8)),
                        with: .color(.blue)
                    )
                }
            }
        }
    }
    
    private func calculateVirtualKeypoints() -> [Int: Keypoint] {
        var virtualKeypoints: [Int: Keypoint] = [:]
        
        // STERNUM (24) = midpoint of shoulders
        if keypoints.count > 5 && keypoints.count > 2 {
            let leftShoulder = keypoints[5]
            let rightShoulder = keypoints[2]
            
            if leftShoulder.confidence > 0.3 && rightShoulder.confidence > 0.3 {
                virtualKeypoints[24] = Keypoint(
                    x: (leftShoulder.x + rightShoulder.x) / 2.0,
                    y: (leftShoulder.y + rightShoulder.y) / 2.0,
                    confidence: min(leftShoulder.confidence, rightShoulder.confidence)
                )
            }
        }
        
        // SACRUM (25) = midpoint of hips
        if keypoints.count > 11 && keypoints.count > 8 {
            let leftHip = keypoints[11]
            let rightHip = keypoints[8]
            
            if leftHip.confidence > 0.3 && rightHip.confidence > 0.3 {
                virtualKeypoints[25] = Keypoint(
                    x: (leftHip.x + rightHip.x) / 2.0,
                    y: (leftHip.y + rightHip.y) / 2.0,
                    confidence: min(leftHip.confidence, rightHip.confidence)
                )
            }
        }
        
        return virtualKeypoints
    }
    
    private func getKeypoint(at index: Int, virtual: [Int: Keypoint]) -> Keypoint {
        if index < keypoints.count {
            return keypoints[index]
        } else if let virtualKeypoint = virtual[index] {
            return virtualKeypoint
        } else {
            return Keypoint(x: 0, y: 0, confidence: 0)
        }
    }
}

// MARK: - Pose Estimator
class PoseEstimator: ObservableObject {
    private var model: MLModel?
    private var inputSize: CGSize = CGSize(width: 224, height: 224)
    private var inputName: String = "image"
    @Published var isModelReady: Bool = false
    
    // Performance logging counters
    private static var perfLogCounter = 0
    
    init(modelPath: String) {
        loadModel(from: modelPath)
    }
    
    private func loadModel(from path: String) {
        print("🔄 Loading model from: \(path)")
        
        let modelURL = URL(fileURLWithPath: path)
        print("📂 Model URL: \(modelURL)")
        
        // Check if it's a .mlpackage that needs compilation
        if path.hasSuffix(".mlpackage") {
            print("📦 .mlpackage detected, compiling to .mlmodelc...")
            compileAndLoadModel(from: modelURL)
        } else {
            // Try direct loading for .mlmodelc files
            loadCompiledModel(from: modelURL)
        }
    }
    
    private func compileAndLoadModel(from url: URL) {
        Task {
            do {
                print("⚙️ Compiling model...")
                let compiledURL = try await MLModel.compileModel(at: url)
                print("✅ Model compiled to: \(compiledURL)")
                
                await MainActor.run {
                    self.loadCompiledModel(from: compiledURL)
                }
            } catch {
                print("❌ Failed to compile model: \(error)")
            }
        }
    }
    
    private func loadCompiledModel(from url: URL) {
        do {
            // Configure for optimal performance
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Use CPU + GPU + Neural Engine
            config.allowLowPrecisionAccumulationOnGPU = true  // Enable FP16 on GPU
            
            self.model = try MLModel(contentsOf: url, configuration: config)
            print("✅ Model loaded successfully with optimal configuration!")
            print("🚀 Compute units: ALL (CPU + GPU + Neural Engine)")
            extractModelInputInfo()
            
            DispatchQueue.main.async {
                self.isModelReady = true
            }
        } catch {
            print("❌ Failed to load compiled model: \(error)")
            DispatchQueue.main.async {
                self.isModelReady = false
            }
        }
    }
    
    private func extractModelInputInfo() {
        guard let model = model else { return }
        
        for (name, description) in model.modelDescription.inputDescriptionsByName {
            if let imageConstraint = description.imageConstraint {
                self.inputName = name
                self.inputSize = CGSize(width: imageConstraint.pixelsWide, height: imageConstraint.pixelsHigh)
                print("📏 Model input detected: \(name) - \(Int(inputSize.width))x\(Int(inputSize.height))")
                return
            }
        }
        
        print("⚠️ Using default input size 224x224")
    }
    
    func getInputSize() -> CGSize {
        return inputSize
    }
    
    func estimatePose(from pixelBuffer: CVPixelBuffer, completion: @escaping ([Keypoint], Double) -> Void) {
        guard let model = model else {
            print("❌ Model is nil!")
            completion([], 0.0)
            return
        }
        
        // Run inference on background queue to avoid blocking main thread
        DispatchQueue.global(qos: .userInteractive).async {
            let preprocessStart = CFAbsoluteTimeGetCurrent()
            
            guard let resizedBuffer = pixelBuffer.resized(to: self.inputSize) else {
                print("❌ Failed to resize pixel buffer to \(self.inputSize)")
                DispatchQueue.main.async {
                    completion([], 0.0)
                }
                return
            }
            
            let preprocessTime = (CFAbsoluteTimeGetCurrent() - preprocessStart) * 1000
            let inferenceStart = CFAbsoluteTimeGetCurrent()
            
            do {
                let input = try MLDictionaryFeatureProvider(dictionary: [self.inputName: MLFeatureValue(pixelBuffer: resizedBuffer)])
                let output = try model.prediction(from: input)
                
                let inferenceTime = (CFAbsoluteTimeGetCurrent() - inferenceStart) * 1000
            
            // Debug: Print all available output features
            let outputFeatureNames = output.featureNames
            
            // Try to find heatmaps with different possible names
            let possibleHeatmapNames = ["heatmaps", "output", "predictions", "keypoint_heatmaps", "pose_heatmaps"]
            var heatmaps: MLMultiArray?
            var usedFeatureName: String?
            
            for featureName in possibleHeatmapNames {
                if let feature = output.featureValue(for: featureName)?.multiArrayValue {
                    heatmaps = feature
                    usedFeatureName = featureName
                    break
                }
            }
            
            guard let heatmaps = heatmaps else {
                print("❌ No heatmaps found in model output")
                print("📋 Available features: \(outputFeatureNames)")
                completion([], 0.0)
                return
            }
            
            
                            let keypoints = self.parseKeypoints(from: heatmaps)
                let avgConfidence = keypoints.map { $0.confidence }.reduce(0, +) / Double(keypoints.count)
            
                // Log performance occasionally to reduce overhead
                PoseEstimator.perfLogCounter += 1
                if PoseEstimator.perfLogCounter % 30 == 0 {
                    print("🎯 Parsed \(keypoints.count) keypoints, avg confidence: \(String(format: "%.3f", avgConfidence))")
                    print("⏱️ Timing - Preprocess: \(String(format: "%.1f", preprocessTime))ms, Inference: \(String(format: "%.1f", inferenceTime))ms")
                }
                
                DispatchQueue.main.async {
                    completion(keypoints, avgConfidence)
                }
                
            } catch {
                print("❌ Prediction error: \(error)")
                DispatchQueue.main.async {
                    completion([], 0.0)
                }
            }
        }
    }
    
    private func parseKeypoints(from heatmaps: MLMultiArray) -> [Keypoint] {
        var keypoints: [Keypoint] = []
        
        let numKeypoints = heatmaps.shape[1].intValue
        let heatmapSize = heatmaps.shape[2].intValue
        
        for keypointIdx in 0..<numKeypoints {
            var maxVal: Double = 0
            var maxX: Int = 0
            var maxY: Int = 0
            
            for y in 0..<heatmapSize {
                for x in 0..<heatmapSize {
                    let index = [0, keypointIdx, y, x] as [NSNumber]
                    let value = heatmaps[index].doubleValue
                    
                    if value > maxVal {
                        maxVal = value
                        maxX = x
                        maxY = y
                    }
                }
            }
            
            let normalizedX = Double(maxX) / Double(heatmapSize - 1)
            let normalizedY = Double(maxY) / Double(heatmapSize - 1)
            
            keypoints.append(Keypoint(x: normalizedX, y: normalizedY, confidence: maxVal))
        }
        
        return keypoints
    }
}

// MARK: - CVPixelBuffer Extension
extension CVPixelBuffer {
    private static var logCounter = 0
    
    func resized(to size: CGSize) -> CVPixelBuffer? {
        // Fast GPU-accelerated resizing using CIImage
        return resizeWithCoreImage(to: size)
    }
    
    private func resizeWithCoreImage(to size: CGSize) -> CVPixelBuffer? {
        let width = Int(size.width)
        let height = Int(size.height)
        
        var resizedBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ] as CFDictionary
        
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &resizedBuffer)
        
        guard status == kCVReturnSuccess, let buffer = resizedBuffer else {
            return nil
        }
        
        // Use CIContext with Metal for GPU acceleration
        let ciContext = CIContext(options: [.useSoftwareRenderer: false])
        let ciImage = CIImage(cvPixelBuffer: self)
        
        // Scale to target size
        let scaleX = CGFloat(width) / CGFloat(CVPixelBufferGetWidth(self))
        let scaleY = CGFloat(height) / CGFloat(CVPixelBufferGetHeight(self))
        let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        // Render to output buffer
        ciContext.render(scaledImage, to: buffer)
        
        return buffer
    }
} 
