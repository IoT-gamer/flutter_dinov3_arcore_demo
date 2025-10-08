package com.example.flutter_dinov3_arcore_demo

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.imgcodecs.Imgcodecs
import java.nio.FloatBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt



class SegmentationProcessor(private val context: Context) {

    companion object {
        private const val PATCH_SIZE = 16
        private const val IMAGE_SIZE = 400 
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null

    private val scope = CoroutineScope(Dispatchers.Default)

    init {
        scope.launch {
            session = createONNXSession()
            println("✅ ONNX Session Initialized in Kotlin")
        }
    }

    private fun createONNXSession(): OrtSession {
        val modelBytes = context.assets.open("dinov3_feature_extractor.onnx").readBytes()
        val options = SessionOptions()

        options.addNnapi() 
        options.addCPU(true)
        options.setOptimizationLevel(OptLevel.ALL_OPT)

        return ortEnv.createSession(modelBytes, options)
    }

    private fun preprocessFrame(rotatedRgbMat: Mat): Pair<FloatBuffer, Map<String, Int>>? {
        if (rotatedRgbMat.rows() == 0 || rotatedRgbMat.cols() == 0) {
            return null
        }
        val hPatches = IMAGE_SIZE / PATCH_SIZE
        var wPatches = (rotatedRgbMat.cols() * IMAGE_SIZE) / (rotatedRgbMat.rows() * PATCH_SIZE)
        if (wPatches % 2 != 0) wPatches -= 1

        val newH = hPatches * PATCH_SIZE
        val newW = wPatches * PATCH_SIZE

        val resizedMat = Mat()
        Imgproc.resize(rotatedRgbMat, resizedMat, Size(newW.toDouble(), newH.toDouble()), 0.0, 0.0, Imgproc.INTER_CUBIC)

        val numPatches = wPatches * hPatches
        val inputTensor = ByteBuffer.allocateDirect(1 * 3 * newH * newW * 4) // 4 bytes per float
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        inputTensor.rewind()

        // Explicitly define the ByteArray size to help the compiler
        val rgbData = ByteArray(newW * newH * 3)
        resizedMat.get(0, 0, rgbData)

        for (c in 0..2) { // R, G, B
            for (h in 0 until newH) {
                for (w in 0 until newW) {
                    val pixelIndex = (h * newW + w) * 3
                    val value = (rgbData[pixelIndex + c].toInt() and 0xFF) / 255.0f
                    inputTensor.put((value - IMAGENET_MEAN[c]) / IMAGENET_STD[c])
                }
            }
        }
        inputTensor.rewind()
        resizedMat.release() // Release memory

        val metadata = mapOf("w_patches" to wPatches, "h_patches" to hPatches, "num_patches" to numPatches)
        return Pair(inputTensor, metadata)
    }

    private fun cosineSimilarity(vec1: FloatArray, vec2: FloatArray): Float {
        var dotProduct = 0.0f
        var mag1 = 0.0f
        var mag2 = 0.0f
        for (i in vec1.indices) {
            dotProduct += vec1[i] * vec2[i]
            mag1 += vec1[i] * vec1[i]
            mag2 += vec2[i] * vec2[i]
        }
        mag1 = sqrt(mag1)
        mag2 = sqrt(mag2)
        return if (mag1 == 0.0f || mag2 == 0.0f) 0.0f else dotProduct / (mag1 * mag2)
    }

    suspend fun performSegmentation(
        frameMat: Mat,
        objectPrototype: FloatArray
    ): Map<String, Any>? {
        val currentSession = this.session ?: return null

        val preprocessResult = preprocessFrame(frameMat)
        if (preprocessResult == null) {
            println("❌ Preprocessing failed for segmentation frame.")
            return null
        }
        val (inputBuffer, metadata) = preprocessResult
        val shape = longArrayOf(1, 3, (metadata["h_patches"]!! * PATCH_SIZE).toLong(), (metadata["w_patches"]!! * PATCH_SIZE).toLong())

        val tensor = OnnxTensor.createTensor(ortEnv, inputBuffer, shape)
        // Explicitly define the map type to help the compiler
        val inputs = mapOf<String, OnnxTensor>("input_image" to tensor)
        
        val outputs = currentSession.run(inputs)
        val featuresTensor = outputs?.get(0) as OnnxTensor
        val testFeatures = featuresTensor.floatBuffer.array()
        
        tensor.close()
        outputs.close()

        val numPatches = metadata["num_patches"]!!
        val featureDim = testFeatures.size / numPatches
        val similarityScores = FloatArray(numPatches)

        for (i in 0 until numPatches) {
            val feature = testFeatures.sliceArray(i * featureDim until (i + 1) * featureDim)
            similarityScores[i] = cosineSimilarity(feature, objectPrototype)
        }

        var finalScores = similarityScores.map { it.toDouble() }

        // Logic for finding the largest connected area
        val wPatches = metadata["w_patches"]!!
        val hPatches = metadata["h_patches"]!!
        val threshold = 0.7f // Threshold to consider a patch as "active"

        val visited = Array(hPatches) { BooleanArray(wPatches) }
        val allComponents = mutableListOf<List<Int>>()

        for (r in 0 until hPatches) {
            for (c in 0 until wPatches) {
                val index = r * wPatches + c
                if (similarityScores[index] > threshold && !visited[r][c]) {
                    // Start a new BFS for a new component
                    val currentComponent = mutableListOf<Int>()
                    val queue = ArrayDeque<Pair<Int, Int>>()

                    queue.add(Pair(r, c))
                    visited[r][c] = true

                    while (queue.isNotEmpty()) {
                        val (currR, currC) = queue.removeFirst()
                        val currIndex = currR * wPatches + currC
                        currentComponent.add(currIndex)

                        // Check 4 neighbors (up, down, left, right)
                        for (dR in -1..1) {
                            for (dC in -1..1) {
                                // Only check direct neighbors, not diagonals
                                if (Math.abs(dR) + Math.abs(dC) != 1) continue

                                val nextR = currR + dR
                                val nextC = currC + dC

                                if (nextR in 0 until hPatches && nextC in 0 until wPatches && !visited[nextR][nextC]) {
                                    val nextIndex = nextR * wPatches + nextC
                                    if (similarityScores[nextIndex] > threshold) {
                                        visited[nextR][nextC] = true
                                        queue.add(Pair(nextR, nextC))
                                    }
                                }
                            }
                        }
                    }
                    allComponents.add(currentComponent)
                }
            }
        }

        // Find the largest component by size
        val largestComponent = allComponents.maxByOrNull { it.size }

        var centroidX = 0f
        var centroidY = 0f

        if (largestComponent != null) {
            val filteredScores = FloatArray(numPatches) { 0.0f }
            var totalX = 0
            var totalY = 0
            for (index in largestComponent) {
                filteredScores[index] = similarityScores[index]
                // Calculate centroid
                val r = index / wPatches // row
                val c = index % wPatches // column
                totalX += c
                totalY += r
            }
            centroidX = totalX.toFloat() / largestComponent.size
            centroidY = totalY.toFloat() / largestComponent.size

            finalScores = filteredScores.map { it.toDouble() }
        } else {
            // If no components are found, return all zeros
            finalScores = FloatArray(numPatches) { 0.0f }.map { it.toDouble() }
        }

        return mapOf(
            "scores" to finalScores,
            "width" to metadata["w_patches"]!!,
            "height" to metadata["h_patches"]!!,
            "centroid_x" to centroidX,
            "centroid_y" to centroidY
        )
    }

    /**
     * Creates a feature vector prototype from a reference RGBA image.
     */
    suspend fun createPrototype(imageBytes: ByteArray): FloatArray? {
        val currentSession = this.session ?: return null

        // Decode image bytes into an OpenCV Mat
        val rgbaMat = Imgcodecs.imdecode(MatOfByte(*imageBytes), Imgcodecs.IMREAD_UNCHANGED)
        if (rgbaMat.channels() != 4) {
            println("❌ Invalid image format. Must be RGBA.")
            return null
        }

        // Split RGBA into RGB and an alpha mask
        val channels = mutableListOf<Mat>()
        Core.split(rgbaMat, channels)
        val rgbMat = Mat()
        val maskMat = channels[3] // Alpha channel is the mask
        Core.merge(channels.subList(0, 3), rgbMat)
        Imgproc.cvtColor(rgbMat, rgbMat, Imgproc.COLOR_BGR2RGB) // Convert BGR (OpenCV default) to RGB

        // Preprocess the image for the model
        val preprocessResult = preprocessFrame(rgbMat)
        if (preprocessResult == null) {
            println("❌ Preprocessing failed for prototype creation.")
            return null
        }
        val (inputBuffer, metadata) = preprocessResult
        val hPatches = metadata["h_patches"]!!
        val wPatches = metadata["w_patches"]!!
        
        // Create a downsized patch mask from the alpha channel
        val resizedMask = Mat()
        Imgproc.resize(maskMat, resizedMask, Size(wPatches.toDouble(), hPatches.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
        val maskData = ByteArray(wPatches * hPatches)
        resizedMask.get(0, 0, maskData)
        val patchMask = maskData.map { (it.toInt() and 0xFF) > 127 }

        // Run inference to get features
        val shape = longArrayOf(1, 3, (hPatches * PATCH_SIZE).toLong(), (wPatches * PATCH_SIZE).toLong())
        val tensor = OnnxTensor.createTensor(ortEnv, inputBuffer, shape)
        val inputs = mapOf("input_image" to tensor)
        val outputs = currentSession.run(inputs)
        val featuresTensor = outputs?.get(0) as OnnxTensor
        val allFeatures = featuresTensor.floatBuffer.array()
        
        tensor.close()
        outputs.close()

        // Average features from foreground patches to create the prototype
        val numPatches = metadata["num_patches"]!!
        val featureDim = allFeatures.size / numPatches
        val foregroundFeatures = mutableListOf<FloatArray>()

        for (i in 0 until numPatches) {
            if (patchMask[i]) {
                foregroundFeatures.add(allFeatures.sliceArray(i * featureDim until (i + 1) * featureDim))
            }
        }
        
        if (foregroundFeatures.isEmpty()) return null

        val objectPrototype = FloatArray(featureDim) { 0.0f }
        for (feature in foregroundFeatures) {
            for (i in 0 until featureDim) {
                objectPrototype[i] += feature[i]
            }
        }
        for (i in 0 until featureDim) {
            objectPrototype[i] /= foregroundFeatures.size
        }
        
        println("✅ Prototype created in Kotlin.")
        rgbaMat.release()
        rgbMat.release()
        maskMat.release()
        resizedMask.release()

        return objectPrototype
    }

    fun close() {
        session?.close()
    }
}