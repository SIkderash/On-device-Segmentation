package com.example.segmentationsam

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import ai.onnxruntime.*
import java.io.File
import java.io.IOException
import java.nio.FloatBuffer

class MainActivity : ComponentActivity() {

    private lateinit var ortEnv: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private lateinit var inputImage: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        try {
            // Initialize ONNX Runtime environment
            ortEnv = OrtEnvironment.getEnvironment()

            // Load ONNX model from assets
            val modelPath = getModelPath("model.onnx")
            ortSession = ortEnv.createSession(modelPath, OrtSession.SessionOptions())
            Log.d("ONNX", "Model loaded successfully")

            // Load image from assets and preprocess it
            inputImage = loadImageFromAssets("image.jpeg")
            val inputTensor = preprocessImage(inputImage)

            // Convert the preprocessed image data into an OnnxTensor
            val onnxTensor = createOnnxTensor(ortEnv, inputTensor)

            // Run inference and get the output
            val output = runInference(onnxTensor)


            Log.d("ONNX_OUTPUT", "Output dimensions: "+ output.toList().toString())

            // Post-process and display the segmented output
            displaySegmentedOutput(output)

        } catch (e: Exception) {
            Log.e("ONNX_ERROR", "Error in ONNX processing: ${e.message}")
        }
    }

    // Load image from assets folder
    private fun loadImageFromAssets(fileName: String): Bitmap {
        return try {
            val inputStream = assets.open(fileName)
            BitmapFactory.decodeStream(inputStream)
        } catch (e: IOException) {
            Log.e("IMAGE_LOAD_ERROR", "Error loading image: ${e.message}")
            throw RuntimeException("Error loading image from assets")
        }
    }

    // Preprocess image for input into the model
    private fun preprocessImage(image: Bitmap): FloatArray {
        val inputWidth = 416
        val inputHeight = 416
        val mean = 127.5f
        val std = 127.5f
        val imgData = FloatArray(1 * 3 * inputWidth * inputHeight)
        val scaledBitmap = Bitmap.createScaledBitmap(image, inputWidth, inputHeight, true)

        var idx = 0
        for (y in 0 until inputHeight) {
            for (x in 0 until inputWidth) {
                val pixel = scaledBitmap.getPixel(x, y)
                // Normalize RGB values to [-1, 1]
                imgData[idx++] = ((pixel shr 16 and 0xFF) - mean) / std // R
                imgData[idx++] = ((pixel shr 8 and 0xFF) - mean) / std  // G
                imgData[idx++] = ((pixel and 0xFF) - mean) / std        // B
            }
        }
        return imgData
    }

    // Create OnnxTensor from the input data
    private fun createOnnxTensor(ortEnv: OrtEnvironment, inputTensor: FloatArray): OnnxTensor {
        val shape = longArrayOf(1, 3, 416, 416)
        val floatBuffer = createFloatBuffer(inputTensor)
        return OnnxTensor.createTensor(ortEnv, floatBuffer, shape)
    }

    // Helper function to create FloatBuffer from a FloatArray
    private fun createFloatBuffer(inputTensor: FloatArray): FloatBuffer {
        val buffer = FloatBuffer.allocate(inputTensor.size)
        buffer.put(inputTensor)
        buffer.rewind()
        return buffer
    }

    // Run the model on the input tensor and get output
    private fun runInference(tensor: OnnxTensor): Array<Any> {
        val results = ortSession.run(mapOf("images" to tensor))
        // Adjusting to handle 3D output (float[][][])
        val output = results[0].value as Array<Array<FloatArray>>  // 3D array: [channels][height][width]

        // Extract the first channel for segmentation (usually first channel is relevant for segmentation)
        val segmentationMap = output[0]  // This is a 2D array [height, width]
        return arrayOf(segmentationMap)
    }

    // Display the segmentation output on the screen
    private fun displaySegmentedOutput(output: Array<Any>) {
        val segmentationMap = output[0] as Array<FloatArray>
        val segmentedBitmap = segmentationMapToBitmap(segmentationMap)

        // Show the segmented image in the ImageView
        val imageView: ImageView = findViewById(R.id.imageView)
        imageView.setImageBitmap(segmentedBitmap)
    }

    // Convert segmentation map (float[][]) into a Bitmap for display
    private fun segmentationMapToBitmap(segmentationMap: Array<FloatArray>): Bitmap {
        val height = segmentationMap.size
        val width = segmentationMap[0].size
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val value = segmentationMap[y][x]

                // Apply threshold for segmentation visualization
                val color = if (value > 0.5f) {
                    0xFF00FF00.toInt()  // Green color for segmented area
                } else {
                    0x00000000  // Transparent for background
                }
                bitmap.setPixel(x, y, color)
            }
        }
        return bitmap
    }

    // Helper function to copy model from assets folder to device storage and return the path
    private fun getModelPath(modelFileName: String): String {
        val file = File(filesDir, modelFileName)
        if (!file.exists()) {
            assets.open(modelFileName).use { input ->
                file.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return file.absolutePath
    }
}
