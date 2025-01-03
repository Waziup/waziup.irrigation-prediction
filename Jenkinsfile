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
    }
}