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
                    // Check if the builder already exists, if not create it
                    def builderExists = sh(script: "docker buildx ls | grep crossbuilder || true", returnStatus: true)
                    if (builderExists != 0) {
                        sh 'docker buildx create --name crossbuilder --use'
                    } else {
                       sh 'docker buildx use crossbuilder'
                    }

                    sh 'docker buildx inspect crossbuilder' // TODO: Verify builder
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