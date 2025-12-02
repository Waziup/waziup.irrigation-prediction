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
        DOCKER_TAG_NAME = 'latest'
        DOCKER_PLATFORM = 'linux/arm64/v8'
        FORMER_IMAGES_DOCKER_ID = ''
        APP_NAME = 'waziup.irrigation-prediction'
        LOCAL_WAZIGATE_IP = 'wazigate-ci.local'
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
                        def result = sh(script: "docker rmi ${formerID}", returnStdout: true, returnStatus: true)
                        if (result != 0) {
                            echo "Error removing old image: ${result}"
                            echo "Failed to remove old image."
                        } else {
                            echo "Successfully removed old image."
                        }
                    }
                    else {
                        echo "No old image to remove."
                    }
                }
            }
        }

        stage('Push to local Gateway & Restart') {
            steps {
                catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                    script {
                        def dockerImage = "${DOCKER_IMAGE_NAME}:${DOCKER_TAG_NAME}"
                        echo "Pushing Docker image to local gateway..."
                        withCredentials([string(credentialsId: 'SSH_PASSWORD_WAZIGATE', variable: 'SSH_PASSWORD_WAZIGATE')]) {
                            sh "docker save ${dockerImage} | gzip | pv | sshpass -p '${SSH_PASSWORD_WAZIGATE}' ssh -o StrictHostKeyChecking=no pi@${LOCAL_WAZIGATE_IP} docker load"
                        }
                        echo "Successfully pushed image ${dockerImage} to local gateway."

                        echo "Copy docker-compose and package.json files to local gateway..."
                        withCredentials([string(credentialsId: 'SSH_PASSWORD_WAZIGATE', variable: 'SSH_PASSWORD_WAZIGATE')]) {
                            sh """
                                # Copy to a temp directory
                                sshpass -p '${SSH_PASSWORD_WAZIGATE}' scp -o StrictHostKeyChecking=no docker-compose.yml pi@${LOCAL_WAZIGATE_IP}:/tmp/docker-compose.yml
                                sshpass -p '${SSH_PASSWORD_WAZIGATE}' scp -o StrictHostKeyChecking=no package.json pi@${LOCAL_WAZIGATE_IP}:/tmp/package.json

                                # Move with sudo on the remote host
                                sshpass -p '${SSH_PASSWORD_WAZIGATE}' ssh -o StrictHostKeyChecking=no pi@${LOCAL_WAZIGATE_IP} '
                                    sudo -S mkdir -p /var/lib/wazigate/apps/${APP_NAME} && \
                                    sudo -S mv /tmp/docker-compose.yml /var/lib/wazigate/apps/${APP_NAME}/docker-compose.yml && \
                                    sudo -S mv /tmp/package.json /var/lib/wazigate/apps/${APP_NAME}/package.json"
                                '
                            """
                        }
                        echo "Successfully copied docker-compose and package.json files to local gateway."
                        
                        echo "Deploying updated container on gateway..."
                        withCredentials([string(credentialsId: 'SSH_PASSWORD_WAZIGATE', variable: 'SSH_PASSWORD_WAZIGATE')]) {
                            sh """
                                sshpass -p '${SSH_PASSWORD_WAZIGATE}' ssh -o StrictHostKeyChecking=no pi@${LOCAL_WAZIGATE_IP} '
                                    cd /var/lib/wazigate/apps/${APP_NAME} && \
                                    docker-compose down && \
                                    docker-compose up -d && \
                                    docker image prune -f
                                '
                            """
                        }
                        echo "Successfully deployed ${dockerImage} to local gateway and cleaned up."   
                    }
                }
            }
        }

        stage('Test Docker Image - Run Unit Tests') {
            steps {
                catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                    script {
                        def service_name = "wazi-app"
                        echo "Running tests on locally deployed Docker image..."
                        withCredentials([string(credentialsId: 'SSH_PASSWORD_WAZIGATE', variable: '	SSH_PASSWORD_WAZIGATE')]) {
                            sh """
                                sshpass -p '${SSH_PASSWORD_WAZIGATE}' ssh -o StrictHostKeyChecking=no pi@${LOCAL_WAZIGATE_IP} '
                                    cd /var/lib/wazigate/apps/${APP_NAME} && \
                                    docker-compose exec ${service_name} python3 -m unittest discover -s tests
                                '
                            """
                        }
                        echo "Successfully deployed ${dockerImage} to local gateway."
                        catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                            echo "Exception during testing: ${e.getMessage()} \nFailed to test deployed image on local gateway."
                            echo "One or more test did failed on local gateway."
                        }
                    }
                }
            }
        }

        stage('Save Docker Image') {
            steps {
                catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                    script {
                        // sh 'rm -f irrigation_prediction_docker_image.tar'
                        sh "docker save ${DOCKER_IMAGE_NAME}:${DOCKER_TAG_NAME} > irrigation_prediction_docker_image.tar"
                        archiveArtifacts artifacts: 'irrigation_prediction_docker_image.tar', fingerprint: true
                    }
                }
            }
        }

        stage('Push to dockerhub'){
            when { expression { params.perform_push_duckerhub } }
            steps {
                catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                    script {
                        def dockerImage = "${DOCKER_IMAGE_NAME}:${DOCKER_TAG_NAME}" // Combines image name and tag
                        retry(2) {
                            sh "docker push ${dockerImage}"
                        }
                        echo "Successfully pushed image ${dockerImage} to Docker Hub."
                    }
                }
            }
        }
    }
}
