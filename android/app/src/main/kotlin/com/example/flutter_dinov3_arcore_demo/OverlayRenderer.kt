package com.example.flutter_dinov3_arcore_demo

import android.content.Context
import android.opengl.GLES20
import com.google.ar.core.examples.java.common.rendering.ShaderUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

class OverlayRenderer {
    private var program: Int = 0
    private val textureId = IntArray(1)
    private var positionAttrib: Int = 0
    private var texCoordAttrib: Int = 0
    private var textureUniform: Int = 0
    
    private val quadCoords: FloatBuffer
    private val quadTexCoords: FloatBuffer = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().put(
        floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f) // Flipped vertically
    )

    init {
        // Full-screen quad coordinates
        quadCoords = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().put(
            floatArrayOf(-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
        )
        quadCoords.position(0)
        quadTexCoords.position(0)
    }

    fun createOnGlThread(context: Context) {
        val vertexShader = ShaderUtil.loadGLShader("Overlay", context, GLES20.GL_VERTEX_SHADER, "shaders/overlay.vert")
        val fragmentShader = ShaderUtil.loadGLShader("Overlay", context, GLES20.GL_FRAGMENT_SHADER, "shaders/overlay.frag")
        
        program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, vertexShader)
        GLES20.glAttachShader(program, fragmentShader)
        GLES20.glLinkProgram(program)
        
        positionAttrib = GLES20.glGetAttribLocation(program, "a_Position")
        texCoordAttrib = GLES20.glGetAttribLocation(program, "a_TexCoord")
        textureUniform = GLES20.glGetUniformLocation(program, "s_Texture")
        
        GLES20.glGenTextures(1, textureId, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId[0])
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST)
    }

    fun update(scores: List<Double>, width: Int, height: Int, threshold: Float) {
        val pixels = ByteBuffer.allocateDirect(width * height * 4).order(ByteOrder.nativeOrder())
        for (score in scores) {
            if (score > threshold) {
                pixels.put(30.toByte())  // Red
                pixels.put(255.toByte()) // Green
                pixels.put(150.toByte()) // Blue
                pixels.put(120.toByte()) // Alpha
            } else {
                pixels.put(0.toByte()).put(0.toByte()).put(0.toByte()).put(0.toByte())
            }
        }
        pixels.rewind()

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId[0])
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pixels)
    }

    fun draw() {
        if (program == 0) return

        GLES20.glUseProgram(program)
        
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId[0])
        GLES20.glUniform1i(textureUniform, 0)
        
        GLES20.glEnableVertexAttribArray(positionAttrib)
        GLES20.glVertexAttribPointer(positionAttrib, 2, GLES20.GL_FLOAT, false, 0, quadCoords)
        
        GLES20.glEnableVertexAttribArray(texCoordAttrib)
        GLES20.glVertexAttribPointer(texCoordAttrib, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        
        GLES20.glDisableVertexAttribArray(positionAttrib)
        GLES20.glDisableVertexAttribArray(texCoordAttrib)
        GLES20.glDisable(GLES20.GL_BLEND)
    }
}