pipeline {
    agent any

    environment {
        DOCKER_IMAGE_NAME = 'waziup/irrigation-prediction:dev'
        DOCKER_PLATFORM = 'linux/arm64/v8'
    }

    stages {
        stage('Buildx Setup') {
            steps {
                script {
                    //Install docker buildx builder
                    sh 'docker run --rm --privileged multiarch/qemu-user-static --reset -p yes'
                    catchError(buildResult: 'SUCCESS', stageResult: 'SUCCESS') {
                        sh 'docker buildx create --name rpibuilder --platform linux/arm64/v8; true'
                    }
                    sh 'docker buildx use rpibuilder'
                    sh 'docker buildx inspect --bootstrap'
                }
            }
        }

       stage('Docker Cross-Build') {
            steps {
                script {
                    sh """
                        docker buildx build \\
                            --platform ${DOCKER_PLATFORM} \\
                            -t ${DOCKER_IMAGE_NAME} \\
                            --no-cache \\
                            --pull \\
                            --build-arg CACHEBUST=\$(date +%s) \\
                            --load .
                    """
                }
            }
        }
        stage('Clean Old Untagged Images') {
            steps {
                script {
                    def imageName = 'waziup/irrigation-prediction' // Define the image name here
                def removeImageCommand = 'docker rmi $(docker images -q --filter "reference=${imageName}:<none>")'


                    try {
                        def result = sh(script: removeImageCommand, returnStdout: true, returnStatus: true)
                        if (result.exitCode != 0) {
                            echo "Error removing untagged images: ${result.output}"
                            error "Failed to remove untagged images."
                        } else {
                            echo "Successfully removed untagged images."
                        }
                    }
                    catch (Exception e) {
                        echo "Exception thrown during image removal: ${e.getMessage()}"
                        error "Failed to remove untagged images due to an exception."
                    }
                }
            }
        }
    }
}