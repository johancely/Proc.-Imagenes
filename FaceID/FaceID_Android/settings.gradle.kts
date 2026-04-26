pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "FaceID_PCA"
include(":app")
include(":opencv")
project(":opencv").projectDir = java.io.File("C:/Users/Johan/OpenCV-android-sdk/OpenCV-android-sdk/sdk")
