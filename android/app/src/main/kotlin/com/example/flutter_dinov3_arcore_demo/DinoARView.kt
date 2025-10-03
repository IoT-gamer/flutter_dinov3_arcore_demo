package com.example.flutter_dinov3_arcore_demo

import android.app.Activity
import android.content.Context
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfig
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.examples.java.common.helpers.CameraPermissionHelper
import com.google.ar.core.examples.java.common.helpers.DisplayRotationHelper
import com.google.ar.core.examples.java.common.rendering.BackgroundRenderer
import com.google.ar.core.examples.java.common.rendering.ObjectRenderer
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.platform.PlatformView
import java.util.concurrent.atomic.AtomicBoolean 
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

class DinoARView(
    private val activity: Activity,
    private val context: Context,
    messenger: BinaryMessenger,
    id: Int,
    private val lifecycle: Lifecycle
) : PlatformView, DefaultLifecycleObserver, MethodChannel.MethodCallHandler, GLSurfaceView.Renderer {

    private var frameCounter = 0
    // This value means we'll attempt to process on the 4th, 8th, 12th frame, etc.
    private val PROCESS_EVERY_NTH_FRAME = 4 

    private val glSurfaceView: GLSurfaceView
    private val displayRotationHelper: DisplayRotationHelper
    private val backgroundRenderer = BackgroundRenderer()
    
    // // Custom Processors and Renderers
    private val segmentationProcessor: SegmentationProcessor
    private val overlayRenderer = OverlayRenderer()
 
    private var session: Session? = null
    private var installRequested = false
    private var shouldConfigureSession = false

    @Volatile // Ensures visibility across threads
    private var isSessionResumed = false

    private var objectPrototype: FloatArray? = null
    private var isSegmenting = false
    private var similarityThreshold = 0.7f
    
    private val mainScope = CoroutineScope(Dispatchers.Main)
    private val processingScope = CoroutineScope(Dispatchers.Default)

    // Add a flag to prevent processing multiple frames at once.
    // Use AtomicBoolean for thread safety.
    private var isProcessingFrame = AtomicBoolean(false)

    private lateinit var yuvMat: Mat
    private lateinit var rgbMat: Mat
    private lateinit var rotatedRgbMat: Mat

    private val objectRenderer = ObjectRenderer()
    private val anchors = mutableListOf<com.google.ar.core.Anchor>()

    @Volatile private var lastCentroid: Pair<Float, Float>? = null
    @Volatile private var patchGridSize: Pair<Int, Int>? = null

    init {
        // // Initialize OpenCV
        if (!OpenCVLoader.initDebug()) {
            Log.e("DinoARView", "FATAL: OpenCV initialization failed.")
        } else {
            Log.d("DinoARView", "OpenCV initialized successfully.")
            // Initialize the Mat objects AFTER the library is loaded
            yuvMat = Mat()
            rgbMat = Mat()
            rotatedRgbMat = Mat()
        }

        glSurfaceView = GLSurfaceView(context).apply {
            setZOrderMediaOverlay(true)
            setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            holder.setFormat(android.graphics.PixelFormat.TRANSLUCENT)

            preserveEGLContextOnPause = true
            setEGLContextClientVersion(2)
            setRenderer(this@DinoARView)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
            setWillNotDraw(false)
        }
        
        displayRotationHelper = DisplayRotationHelper(context)
        segmentationProcessor = SegmentationProcessor(context)

        val methodChannel = MethodChannel(messenger, "dino_ar_channel")
        methodChannel.setMethodCallHandler(this)
        lifecycle.addObserver(this)
    }

    // --- PlatformView and Lifecycle Methods ---
    override fun getView(): View = glSurfaceView
    
    override fun dispose() {
        lifecycle.removeObserver(this)
        session?.close()
        session = null
        segmentationProcessor.close()
    }

    override fun onResume(owner: LifecycleOwner) {
        if (!CameraPermissionHelper.hasCameraPermission(activity)) {
            CameraPermissionHelper.requestCameraPermission(activity)
            return
        }
        
        glSurfaceView.onResume()
        displayRotationHelper.onResume()
    }

    override fun onPause(owner: LifecycleOwner) {
        isSessionResumed = false

        if (session != null) {
            displayRotationHelper.onPause()
            glSurfaceView.onPause()
            // session!!.pause()
            session?.close()
            session = null
        }
    }

    // --- MethodChannel Handler ---
    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "createPrototype" -> {
                val bytes = call.argument<ByteArray>("bytes")
                if (bytes != null) {
                    processingScope.launch {
                        val prototype = segmentationProcessor.createPrototype(bytes)
                        if (prototype != null) {
                            objectPrototype = prototype
                            mainScope.launch {
                                Toast.makeText(context, "Prototype created successfully!", Toast.LENGTH_SHORT).show()
                                result.success(true)
                            }
                        } else {
                            mainScope.launch {
                            Toast.makeText(context, "Failed to create prototype.", Toast.LENGTH_LONG).show()
                            result.success(false)
                            }
                        }
                    }
                } else {
                    result.error("INVALID_ARGS", "Byte array is null", null)
                }
            }
            "toggleSegmentation" -> {
                isSegmenting = !isSegmenting
                result.success(isSegmenting)
            }
            "setThreshold" -> {
                val threshold = call.argument<Double>("threshold")
                if (threshold != null) {
                    similarityThreshold = threshold.toFloat()
                    result.success(null)
                } else {
                    result.error("INVALID_ARGS", "Threshold is null", null)
                }
            }
            "placeObjectAtCentroid" -> {
                val currentCentroid = lastCentroid
                val currentGridSize = patchGridSize
                val currentSession = session

                if (currentCentroid == null || currentGridSize == null || currentSession == null) {
                    result.success(false)
                    return
                }

                // Convert patch coordinates to screen coordinates
                val (patchWidth, patchHeight) = currentGridSize
                val (centroidX, centroidY) = currentCentroid

                val viewWidth = glSurfaceView.width.toFloat()
                val viewHeight = glSurfaceView.height.toFloat()

                val screenX = (centroidX / patchWidth) * viewWidth
                val screenY = (centroidY / patchHeight) * viewHeight

                // Perform hit test on the GL thread
                glSurfaceView.queueEvent {
                    try {
                        val frame = currentSession.update()
                        val hitResults = frame.hitTest(screenX, screenY)

                        if (hitResults.isNotEmpty()) {
                            val hit = hitResults.firstOrNull {
                                // Check if the hit is on a plane OR an instant placement point
                                val trackable = it.trackable
                                (trackable is com.google.ar.core.Plane && trackable.isPoseInPolygon(it.hitPose)) ||
                                (trackable is com.google.ar.core.Point && trackable.orientationMode == com.google.ar.core.Point.OrientationMode.ESTIMATED_SURFACE_NORMAL)
                            }

                            if (hit != null) {
                                // Limit the number of anchors
                                if (anchors.size >= 20) {
                                    anchors[0].detach()
                                    anchors.removeAt(0)
                                }
                                anchors.add(hit.createAnchor())
                                mainScope.launch { result.success(true) }
                            } else {
                                mainScope.launch { result.success(false) }
                            }
                        } else {
                            mainScope.launch { result.success(false) }
                        }
                    } catch (e: Exception) {
                        Log.e("DinoARView", "Error during hit test", e)
                        mainScope.launch { result.success(false) }
                    }
                }
            }

            else -> result.notImplemented()
        }
    }
    
    // --- GLSurfaceView.Renderer Methods ---

    override fun onSurfaceCreated(gl: javax.microedition.khronos.opengles.GL10?, config: javax.microedition.khronos.egl.EGLConfig?) {
        GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1.0f)
        
        try {
            backgroundRenderer.createOnGlThread(context)
            overlayRenderer.createOnGlThread(context)
            objectRenderer.createOnGlThread(context, "models/andy.obj", "models/andy.png")
            objectRenderer.setMaterialProperties(0.0f, 2.0f, 0.5f, 6.0f)

            // Create the ARCore session here, after the GL surface and renderers are ready.
            if (session == null) {
                when (ArCoreApk.getInstance().requestInstall(activity, !installRequested)) {
                    ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                        installRequested = true
                        return
                    }
                    ArCoreApk.InstallStatus.INSTALLED -> {}
                    else -> {
                        // Handle other statuses if necessary
                        return
                    }
                }
                
                session = Session(context)
                val arConfig = Config(session)
                arConfig.focusMode = Config.FocusMode.AUTO
                arConfig.instantPlacementMode = Config.InstantPlacementMode.LOCAL_Y_UP
                session?.configure(arConfig)
            }

            // Now that the session is created, immediately resume it.
            session?.resume()
            isSessionResumed = true

        } catch (e: Exception) {
            Log.e("DinoARView", "Failed to create renderer or AR session", e)
        }
    }

    override fun onSurfaceChanged(gl: javax.microedition.khronos.opengles.GL10?, width: Int, height: Int) {
        displayRotationHelper.onSurfaceChanged(width, height)
        GLES20.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: javax.microedition.khronos.opengles.GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        if (session == null || !isSessionResumed) {
            return
        }
        
        displayRotationHelper.updateSessionIfNeeded(session)

        try {
            session!!.setCameraTextureName(backgroundRenderer.textureId)
   
            val frame = session!!.update()
            val camera = frame.camera
            
            backgroundRenderer.draw(frame)

            // Get projection matrix.
            val projmtx = FloatArray(16)
            camera.getProjectionMatrix(projmtx, 0, 0.1f, 100.0f)

            // Get camera matrix.
            val viewmtx = FloatArray(16)
            camera.getViewMatrix(viewmtx, 0)

            // Get lighting information for realistic rendering
            val colorCorrectionRgba = FloatArray(4)
            frame.lightEstimate.getColorCorrection(colorCorrectionRgba, 0)

            // Create a temporary matrix for each object
            val modelMatrix = FloatArray(16)

            // Draw all placed objects
            for (anchor in anchors) {
                if (anchor.trackingState == TrackingState.TRACKING) {
                    // Get the current pose of an Anchor in world space.
                    anchor.pose.toMatrix(modelMatrix, 0)

                    // Update and draw the model.
                    // Adjust the scale-factoras needed for your model.
                    objectRenderer.updateModelMatrix(modelMatrix, 1.0f)
                    objectRenderer.draw(viewmtx, projmtx, colorCorrectionRgba)
                }
            }

            frameCounter++

            val shouldProcessThisFrame = 
                camera.trackingState == TrackingState.TRACKING &&
                isSegmenting &&
                objectPrototype != null &&
                !isProcessingFrame.get() &&
                (frameCounter % PROCESS_EVERY_NTH_FRAME == 0)

            if (shouldProcessThisFrame) {

                frame.acquireCameraImage().use { image ->
                    if (image == null) return@use

                    isProcessingFrame.set(true)

                    val requiredHeight = image.height + image.height / 2
                    if (yuvMat.empty() || yuvMat.rows() != requiredHeight || yuvMat.cols() != image.width) {
                        yuvMat.create(requiredHeight, image.width, CvType.CV_8UC1)
                    }


                    imageToYuvMat(image, yuvMat)
                    
                    Imgproc.cvtColor(yuvMat, rgbMat, Imgproc.COLOR_YUV2RGB_I420)
                    Core.rotate(rgbMat, rotatedRgbMat, Core.ROTATE_90_CLOCKWISE)

                    // --- CROP TO MATCH VIEW ASPECT RATIO ---
                    // This ensures the segmentation input matches the view's aspect ratio
                    val viewWidth = glSurfaceView.width
                    val viewHeight = glSurfaceView.height
                    val imageWidth = rotatedRgbMat.cols()
                    val imageHeight = rotatedRgbMat.rows()

                    val viewAspectRatio = viewWidth.toFloat() / viewHeight.toFloat()
                    val imageAspectRatio = imageWidth.toFloat() / imageHeight.toFloat()

                    val croppedMat: Mat
                    if (imageAspectRatio > viewAspectRatio) {
                        // Image is wider than the view (shouldn't happen in portrait, but good practice)
                        val newWidth = (imageHeight * viewAspectRatio).toInt()
                        val x = (imageWidth - newWidth) / 2
                        croppedMat = Mat(rotatedRgbMat, org.opencv.core.Rect(x, 0, newWidth, imageHeight))
                    } else {
                        // Image is taller than the view (common case)
                        val newHeight = (imageWidth / viewAspectRatio).toInt()
                        val y = (imageHeight - newHeight) / 2
                        croppedMat = Mat(rotatedRgbMat, org.opencv.core.Rect(0, y, imageWidth, newHeight))
                    }                  

                    processingScope.launch {
                        try {

                            val result = segmentationProcessor.performSegmentation(croppedMat, objectPrototype!!)
                            
                            if (result != null) {

                                // Store the centroid and grid size for later use
                                val centroidX = result["centroid_x"] as Float
                                val centroidY = result["centroid_y"] as Float
                                if (centroidX > 0 && centroidY > 0) {
                                    lastCentroid = Pair(centroidX, centroidY)
                                } else {
                                    lastCentroid = null // No object found
                                }
                                patchGridSize = Pair(result["width"] as Int, result["height"] as Int)
                                    
                                mainScope.launch {
                                    val scores = result["scores"] as List<Double>
                                    val width = result["width"] as Int
                                    val height = result["height"] as Int
                                    glSurfaceView.queueEvent {
                                        overlayRenderer.update(scores, width, height, similarityThreshold)
                                    }
                                }
                            }
                        } finally {
                            isProcessingFrame.set(false)
                        }
                    }

                    // NOTE: yuvMat, rgbMat, and rotatedRgbMat are NOT released here.
                    // They are class properties that will be reused in the next frame.

                }
            }
            
            if (isSegmenting) {
                overlayRenderer.draw()
            }

        } catch (t: Throwable) {
            Log.e("DinoARView", "Exception on the OpenGL thread", t)
        }
    }
    
    // --- Helper Methods ---
    private fun configureSession() {
        val config = Config(session)
        config.focusMode = Config.FocusMode.AUTO
        
        
        session?.configure(config)
    }
    
    /**
     * Converts an android.media.Image (in YUV_420_888 format) to an OpenCV Mat.
     *
     * @param image The input image from the camera.
     * @param outMat The pre-allocated Mat object where the YUV data will be written.
     */
    private fun imageToYuvMat(image: android.media.Image, outMat: Mat) {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val yuvBytes = ByteArray(ySize + uSize + vSize)

        // Interleave the V and U planes into the byte array
        yBuffer.get(yuvBytes, 0, ySize)
        vBuffer.get(yuvBytes, ySize, vSize)
        uBuffer.get(yuvBytes, ySize + vSize, uSize)

        // Put the byte array into the pre-allocated Mat object
        outMat.put(0, 0, yuvBytes)
    }
}