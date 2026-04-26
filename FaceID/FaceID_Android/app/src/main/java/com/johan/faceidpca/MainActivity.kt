package com.johan.faceidpca

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.JavaCameraView
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import java.io.File
import java.io.FileOutputStream
import java.util.ArrayDeque

// Actividad principal: coordina camara, captura, entrenamiento PCA y verificacion en vivo.
class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    companion object {
        private const val TAG = "FaceID_PCA"
        private const val CAMERA_PERMISSION_CODE = 100
        private const val MAX_CAPTURES = 10
        private const val MIN_CAPTURES = 5
        private const val PREDICTION_WINDOW = 10
        private const val MIN_MATCHES = 5
    }

    // Estas funciones llaman al codigo C++ donde estan OpenCV, PCA y distancia euclidiana.
    private external fun nativeInitDetector(cascadePath: String): Boolean
    private external fun nativeCaptureFace(matAddr: Long, savePath: String, index: Int): Boolean
    private external fun nativeTrainPCA(trainListPath: String, dataDir: String): Boolean
    private external fun nativeRecognizeFace(matAddr: Long, dataDir: String): String

    private lateinit var cameraView: CameraBridgeViewBase
    private lateinit var tvStatus: TextView
    private lateinit var tvIdentity: TextView
    private lateinit var tvCaptureCount: TextView
    private lateinit var btnCapture: Button
    private lateinit var btnTrain: Button
    private lateinit var btnRecognize: Button
    private lateinit var btnTakePhoto: Button

    private enum class AppMode { IDLE, CAPTURING, RECOGNIZING }

    @Volatile private var currentMode = AppMode.IDLE
    @Volatile private var shouldCapture = false
    private var captureCount = 0
    private var nativeReady = false
    private lateinit var appDataDir: String
    private val recentPredictions = ArrayDeque<String>()

    // Punto de entrada de la pantalla. Prepara UI, carpetas, librerias y permisos.
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()
        prepareFolders()
        nativeReady = loadNativeLibraries() && prepareDetector()
        prepareCamera()
        setupButtons()

        if (!nativeReady) {
            setStatus("Error: OpenCV o el detector no iniciaron. Revisa Logcat.")
            setButtonsEnabled(false)
            return
        }

        requestCameraOrStart()
    }

    // Enlaza variables Kotlin con widgets del layout.
    private fun bindViews() {
        cameraView = findViewById(R.id.camera_view)
        tvStatus = findViewById(R.id.tv_status)
        tvIdentity = findViewById(R.id.tv_identity)
        tvCaptureCount = findViewById(R.id.tv_capture_count)
        btnCapture = findViewById(R.id.btn_capture)
        btnTrain = findViewById(R.id.btn_train)
        btnRecognize = findViewById(R.id.btn_recognize)
        btnTakePhoto = findViewById(R.id.btn_take_photo)
    }

    // Crea estructura local de trabajo y copia clasificadores Haar desde assets.
    private fun prepareFolders() {
        appDataDir = filesDir.absolutePath
        File("$appDataDir/faces/temp").mkdirs()
        File("$appDataDir/data").mkdirs()
        File("$appDataDir/list").mkdirs()
        copyAssetToInternal("haarcascade_frontalface_default.xml")
        copyAssetToInternal("haarcascade_eye.xml")
    }

    // Carga OpenCV y la libreria nativa donde vive PCA y reconocimiento.
    private fun loadNativeLibraries(): Boolean {
        return try {
            if (!OpenCVLoader.initLocal()) {
                Log.e(TAG, "OpenCVLoader.initLocal() fallo")
                return false
            }
            System.loadLibrary("faceid_pca")
            true
        } catch (error: UnsatisfiedLinkError) {
            Log.e(TAG, "No se pudo cargar una libreria nativa", error)
            false
        }
    }

    // Inicializa detector de rostro/ojos en C++ usando los XML locales.
    private fun prepareDetector(): Boolean {
        return try {
            nativeInitDetector("$appDataDir/haarcascade_frontalface_default.xml")
        } catch (error: Throwable) {
            Log.e(TAG, "No se pudo iniciar el detector", error)
            false
        }
    }

    // Configura la vista de OpenCV para usar camara frontal y resolucion manejable.
    private fun prepareCamera() {
        cameraView.visibility = SurfaceView.VISIBLE
        cameraView.setCvCameraViewListener(this)
        cameraView.setMaxFrameSize(1280, 720)
        if (cameraView is JavaCameraView) {
            (cameraView as JavaCameraView).setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT)
        }
    }

    // Conecta acciones de botones con el flujo de captura-entrenamiento-reconocimiento.
    private fun setupButtons() {
        btnCapture.setOnClickListener { startCaptureMode() }
        btnTakePhoto.setOnClickListener { requestSingleCapture() }
        btnTrain.setOnClickListener { trainModel() }
        btnRecognize.setOnClickListener { startRecognitionMode() }
    }

    // Reinicia estado para capturar dataset nuevo del usuario.
    private fun startCaptureMode() {
        currentMode = AppMode.CAPTURING
        recentPredictions.clear()
        captureCount = 0
        clearOldCaptures()
        shouldCapture = false
        tvIdentity.text = ""
        tvCaptureCount.text = "Fotos: 0/$MAX_CAPTURES"
        tvCaptureCount.visibility = View.VISIBLE
        btnTakePhoto.visibility = View.VISIBLE
        setStatus("Modo captura: toma entre $MIN_CAPTURES y $MAX_CAPTURES fotos del rostro.")
    }

    // Marca que el siguiente frame valido debe guardarse como muestra.
    private fun requestSingleCapture() {
        if (currentMode != AppMode.CAPTURING) return
        if (captureCount >= MAX_CAPTURES) return
        shouldCapture = true
        setStatus("Capturando foto ${captureCount + 1}...")
    }

    // Genera lista de entrenamiento y llama al entrenamiento PCA nativo.
    private fun trainModel() {
        if (captureCount < MIN_CAPTURES) {
            toast("Captura al menos $MIN_CAPTURES fotos antes de entrenar.")
            return
        }

        currentMode = AppMode.IDLE
        btnTakePhoto.visibility = View.GONE
        tvCaptureCount.visibility = View.GONE
        setStatus("Entrenando PCA...")
        tvIdentity.text = ""

        generateTrainList()
        val success = nativeTrainPCA("$appDataDir/list/train_list.txt", appDataDir)
        if (success) {
            setStatus("Entrenamiento terminado. Puedes iniciar reconocimiento.")
            toast("Modelo PCA entrenado")
        } else {
            setStatus("No se pudo entrenar. Verifica que las fotos tengan rostro.")
            toast("Error entrenando el modelo")
        }
    }

    // Activa modo de verificacion en tiempo real si ya existe un modelo entrenado.
    private fun startRecognitionMode() {
        val meanFile = File("$appDataDir/data/mean.txt")
        if (!meanFile.exists()) {
            toast("Primero captura fotos y entrena el modelo.")
            return
        }

        currentMode = AppMode.RECOGNIZING
        recentPredictions.clear()
        btnTakePhoto.visibility = View.GONE
        tvCaptureCount.visibility = View.GONE
        setStatus("Reconocimiento activo. Mira a la camara.")
    }

    // Callback de OpenCV cuando la camara arranca.
    override fun onCameraViewStarted(width: Int, height: Int) {
        setStatus("Camara iniciada. Lista para capturar.")
    }

    // Callback de OpenCV cuando la camara se detiene.
    override fun onCameraViewStopped() = Unit

    // Procesa cada frame: captura o reconoce segun el modo actual.
    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        val rgba = inputFrame.rgba()

        when (currentMode) {
            AppMode.CAPTURING -> captureFrameIfRequested(rgba)
            AppMode.RECOGNIZING -> recognizeFrame(rgba)
            AppMode.IDLE -> Unit
        }

        return rgba
    }

    // Guarda una foto cuando el usuario lo pide; si hay rostro, tambien crea variaciones.
    private fun captureFrameIfRequested(rgba: Mat) {
        if (!shouldCapture || captureCount >= MAX_CAPTURES) return
        shouldCapture = false

        val success = nativeCaptureFace(rgba.nativeObjAddr, "$appDataDir/faces/temp", captureCount + 1)
        runOnUiThread {
            if (success) {
                captureCount++
                tvCaptureCount.text = "Fotos: $captureCount/$MAX_CAPTURES"
                setStatus("Foto $captureCount guardada. Tambien se crearon variaciones.")
                if (captureCount >= MAX_CAPTURES) {
                    btnTakePhoto.visibility = View.GONE
                    currentMode = AppMode.IDLE
                    setStatus("Capturas completas. Presiona Entrenar PCA.")
                }
            } else {
                setStatus("No detecte un rostro claro. Intenta de nuevo con mejor luz.")
            }
        }
    }

    // Llama al verificador nativo y actualiza etiqueta visual (aceptado/rechazado).
    private fun recognizeFrame(rgba: Mat) {
        val rawFaceId = nativeRecognizeFace(rgba.nativeObjAddr, appDataDir)
        val faceID = getStableIdentity(rawFaceId)
        runOnUiThread {
            if (faceID != "None" && faceID.isNotBlank()) {
                tvIdentity.text = "Identificado: $faceID"
                tvIdentity.setTextColor(0xFF00C853.toInt())
            } else {
                tvIdentity.text = "Rostro no reconocido"
                tvIdentity.setTextColor(0xFFD50000.toInt())
            }
        }
    }

    // Filtro temporal: evita aceptar por un solo frame aislado.
    // Solo muestra identidad si se repite suficientes veces en la ventana.
    private fun getStableIdentity(rawId: String): String {
        val normalized = if (rawId.isBlank()) "None" else rawId
        recentPredictions.addLast(normalized)
        if (recentPredictions.size > PREDICTION_WINDOW) {
            recentPredictions.removeFirst()
        }

        var bestId = "None"
        var bestCount = 0
        val counts = HashMap<String, Int>()
        for (id in recentPredictions) {
            val next = (counts[id] ?: 0) + 1
            counts[id] = next
            if (next > bestCount) {
                bestCount = next
                bestId = id
            }
        }
        return if (bestId != "None" && bestCount >= MIN_MATCHES) bestId else "None"
    }

    // Copia recursos internos (XML de cascadas) desde assets al almacenamiento privado.
    private fun copyAssetToInternal(filename: String) {
        val outFile = File(appDataDir, filename)
        if (outFile.exists()) return

        assets.open(filename).use { input ->
            FileOutputStream(outFile).use { output -> input.copyTo(output) }
        }
    }

    // Crea train_list.txt en formato "ID;ruta", consumido por el C++.
    private fun generateTrainList() {
        val tempDir = File("$appDataDir/faces/temp")
        val images = tempDir.listFiles()
            ?.filter { it.extension.lowercase() in setOf("bmp", "jpg", "jpeg", "png") }
            ?.sortedBy { it.name }
            .orEmpty()

        // Todas las variaciones pertenecen a la misma persona registrada.
        val lines = images.joinToString(separator = "\n", postfix = "\n") { file ->
            "Persona;${file.absolutePath}"
        }
        File("$appDataDir/list/train_list.txt").writeText(lines)
    }

    // Limpia capturas viejas para no mezclar datasets de sesiones anteriores.
    private fun clearOldCaptures() {
        File("$appDataDir/faces/temp").listFiles()?.forEach { file ->
            if (file.isFile) file.delete()
        }
    }

    // Pide permiso de camara o arranca directamente si ya fue concedido.
    private fun requestCameraOrStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE)
        }
    }

    // Habilita stream de camara en OpenCV.
    private fun startCamera() {
        cameraView.setCameraPermissionGranted()
        cameraView.enableView()
    }

    // Resultado del cuadro de permisos Android.
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            setStatus("Permiso de camara denegado.")
            toast("La app necesita la camara para funcionar")
        }
    }

    // Reanuda camara al volver a foreground.
    override fun onResume() {
        super.onResume()
        if (::cameraView.isInitialized && nativeReady && hasCameraPermission()) startCamera()
    }

    // Libera camara al pausar actividad.
    override fun onPause() {
        if (::cameraView.isInitialized) cameraView.disableView()
        super.onPause()
    }

    // Libera camara al destruir actividad.
    override fun onDestroy() {
        if (::cameraView.isInitialized) cameraView.disableView()
        super.onDestroy()
    }

    // Helper para consultar permiso actual de camara.
    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }

    // Habilita/deshabilita todos los botones de control.
    private fun setButtonsEnabled(enabled: Boolean) {
        btnCapture.isEnabled = enabled
        btnTrain.isEnabled = enabled
        btnRecognize.isEnabled = enabled
        btnTakePhoto.isEnabled = enabled
    }

    // Muestra texto de estado para guiar al usuario.
    private fun setStatus(message: String) {
        tvStatus.text = message
    }

    // Mensaje corto en pantalla.
    private fun toast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}
