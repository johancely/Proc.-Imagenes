plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.johan.faceidpca"
    compileSdk = 34
    ndkVersion = "28.0.13004108"

    defaultConfig {
        applicationId = "com.johan.faceidpca"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++14 -frtti -fexceptions"
                arguments += "-DANDROID_STL=c++_static"
                arguments += "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=16384"
            }
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
            pickFirsts.add("**/libopencv_java4.so")
            pickFirsts.add("**/libc++_shared.so")
        }
        resources {
            excludes += "/lib/**"
        }
    }

    androidResources {
        noCompress += "so"
    }

    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    // ============================================================
    // IMPORTANTE: Cambia esta ruta al directorio donde descargaste
    // el OpenCV Android SDK (descomprimido)
    // ============================================================
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation(project(":opencv"))
}
