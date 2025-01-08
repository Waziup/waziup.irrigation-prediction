pipeline {
    agent any
    parameters {
        booleanParam(name: 'perform_push_duckerhub', defaultValue: false, description: 'Set true to push to dockerhub.')
    }
    options {
        timeout(time: 1, unit: 'HOURS')
    }
    environment {
        DOCKER_IMAGE_NAME = 'waziup/irrigation-prediction'
        DOCKER_TAG_NAME = 'dev'
        DOCKER_PLATFORM = 'linux/arm64/v8'
        FORMER_IMAGES_DOCKER_ID = ''
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
                    formerImagesDockerID = sh(
                        script: "docker images -q --filter reference=${DOCKER_IMAGE_NAME}:${DOCKER_TAG_NAME}",
                        returnStdout: true
                    ).trim()

                    if(formerImagesDockerID) {
                        echo "Saving image id ${formerImagesDockerID} before rebuilding. After the buildx cmd succeeded, it will be deleted."
                    } else {
                        echo "No existing images to remove. No image id will be saved."
                    }
                    sh """
                        docker buildx build \\
                            --platform ${DOCKER_PLATFORM} \\
                            -t ${DOCKER_IMAGE_NAME}:${DOCKER_TAG_NAME} \\
                            --no-cache \\
                            --pull \\
                            --build-arg CACHEBUST=\$(date +%s) \\
                            --load .
                    """
                     // Stash the ID for use in later stages
                    writeFile file: 'former_image_id.txt', text: formerImagesDockerID
                    stash name: 'former_image_id', includes: 'former_image_id.txt'
                }
            }
        }

        stage('Clean Old Untagged Images') {
            steps {
                script {
                    // Unstash the ID from the previous stage
                    unstash 'former_image_id'
                    def formerID = readFile('former_image_id.txt').trim()

                    if(formerID) {
                        try {
                            def result = sh(script: "docker rmi ${formerID}", returnStdout: true, returnStatus: true)
                            if (result != 0) {
                                echo "Error removing old image: ${result}"
                                error "Failed to remove old image."
                            } else {
                                echo "Successfully removed old image."
                            }
                        }
                        catch (Exception e) {
                            echo "Exception thrown during image removal: ${e.getMessage()}"
                            error "Failed to remove old image due to an exception."
                        }
                    }
                    else {
                        echo "No old image to remove."
                    }
                }
            }
        }

        stage('Save Docker Image') {
            steps {
                // sh 'rm -f irrigation_prediction_docker_image.tar'
                sh "docker save ${DOCKER_IMAGE_NAME}:${DOCKER_TAG_NAME} > irrigation_prediction_docker_image.tar"
                archiveArtifacts artifacts: 'irrigation_prediction_docker_image.tar', fingerprint: true
            }
        }

        stage('Push to dockerhub'){
            when { expression { params.perform_push_duckerhub } }
            steps {
                script {
                    def dockerImage = "${DOCKER_IMAGE_NAME}:${DOCKER_TAG_NAME}" // Combines image name and tag
                    try {
                        sh "docker push ${dockerImage}"
                        echo "Successfully pushed image ${dockerImage} to Docker Hub."
                    } catch (Exception e) {
                       echo "Exception while pushing image: ${e.getMessage()}"
                       error "Failed to push image to Docker Hub."
                    }
                }
            }
        }
    }
}