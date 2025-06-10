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
        if lowercased.contains("dinov2") && lowercased.contains("lora") {
            return "DINOv2 LoRA"
        } else if lowercased.contains("dinov2") {
            return "DINOv2"
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
    @Environment(\.dismiss) private var dismiss
    
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
                        
                        InfoChip(text: "FPS: \(String(format: "%.1f", 1000.0 / max(inferenceTime, 1)))")
                        
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
        
        // Process at most 10 FPS to avoid overwhelming the system
        if currentTime - lastProcessTime < 0.1 {
            return
        }
        lastProcessTime = currentTime
        
        print("🎥 Processing frame...")
        
        estimator.estimatePose(from: pixelBuffer) { keypoints, confidence in
            let endTime = CFAbsoluteTimeGetCurrent()
            let timeElapsed = (endTime - startTime) * 1000
            
            Task { @MainActor in
                self.currentKeypoints = keypoints
                self.avgConfidence = confidence
                self.inferenceTime = timeElapsed
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
                self.setupCaptureSession()
                self.captureSession?.startRunning()
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
        let session = AVCaptureSession()
        session.sessionPreset = .medium
        
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: currentCameraPosition),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            return
        }
        
        if session.canAddInput(input) {
            session.addInput(input)
            currentInput = input
        }
        
        videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }
        
        updateCameraOrientation()
        
        DispatchQueue.main.async {
            self.captureSession = session
        }
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
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
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
            self.model = try MLModel(contentsOf: url)
            print("✅ Model loaded successfully!")
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
        
        guard let resizedBuffer = pixelBuffer.resized(to: inputSize) else {
            print("❌ Failed to resize pixel buffer to \(inputSize)")
            completion([], 0.0)
            return
        }
        
        do {
            let input = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: resizedBuffer)])
            let output = try model.prediction(from: input)
            
            // Debug: Print all available output features
            let outputFeatureNames = output.featureNames
            print("🔍 Available output features: \(outputFeatureNames)")
            
            // Try to find heatmaps with different possible names
            let possibleHeatmapNames = ["heatmaps", "output", "predictions", "keypoint_heatmaps", "pose_heatmaps"]
            var heatmaps: MLMultiArray?
            var usedFeatureName: String?
            
            for featureName in possibleHeatmapNames {
                if let feature = output.featureValue(for: featureName)?.multiArrayValue {
                    heatmaps = feature
                    usedFeatureName = featureName
                    print("✅ Found heatmaps at feature: \(featureName)")
                    break
                }
            }
            
            guard let heatmaps = heatmaps else {
                print("❌ No heatmaps found in model output")
                print("📋 Available features: \(outputFeatureNames)")
                completion([], 0.0)
                return
            }
            
            print("📊 Heatmaps shape: \(heatmaps.shape) (using feature: \(usedFeatureName ?? "unknown"))")
            
            let keypoints = parseKeypoints(from: heatmaps)
            let avgConfidence = keypoints.map { $0.confidence }.reduce(0, +) / Double(keypoints.count)
            
            print("🎯 Parsed \(keypoints.count) keypoints, avg confidence: \(String(format: "%.3f", avgConfidence))")
            
            completion(keypoints, avgConfidence)
            
        } catch {
            print("❌ Prediction error: \(error)")
            completion([], 0.0)
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
    func resized(to size: CGSize) -> CVPixelBuffer? {
        let width = Int(size.width)
        let height = Int(size.height)
        
        let originalWidth = CVPixelBufferGetWidth(self)
        let originalHeight = CVPixelBufferGetHeight(self)
        
        print("📏 Resizing from \(originalWidth)x\(originalHeight) to \(width)x\(height)")
        
        var resizedBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &resizedBuffer)
        
        guard status == kCVReturnSuccess, let buffer = resizedBuffer else { 
            print("❌ Failed to create CVPixelBuffer with status: \(status)")
            return nil 
        }
        
        CVPixelBufferLockBaseAddress(self, .readOnly)
        CVPixelBufferLockBaseAddress(buffer, [])
        
        defer {
            CVPixelBufferUnlockBaseAddress(self, .readOnly)
            CVPixelBufferUnlockBaseAddress(buffer, [])
        }
        
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else { return nil }
        
        let ciImage = CIImage(cvPixelBuffer: self)
        let ciContext = CIContext()
        
        if let cgImage = ciContext.createCGImage(ciImage, from: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(self), height: CVPixelBufferGetHeight(self))) {
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        
        return buffer
    }
} 
