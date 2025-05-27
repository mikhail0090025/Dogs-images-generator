pipeline {
    agent any
    parameters {

    }
    stages {
        stage('Start message') {
            steps {
                echo 'Start of the pipeline'
            }
        }
        stage('Checkout') {
            steps {
                git url: 'https://github.com/mikhail0090025/Dogs-images-generator', branch: 'main'
            }
        }
        stage('Build Docker Images') {
            steps {
                sh '''
                    docker-compose build
                '''
            }
        }
        stage('Run Containers') {
            steps {
                sh '''
                    docker-compose up -d
                '''
            }
        }
        stage('Save Logs') {
            steps {
                sh '''
                    docker-compose logs neural_net > neural_net.log
                '''
                archiveArtifacts artifacts: 'neural_net.log', allowEmptyArchive: true
            }
        }
    }
    post {
        always {
            sh '''
                docker-compose down
            '''
            echo 'Pipeline finished!'
        }
        success {
            echo 'GAN training completed successfully!'
        }
        failure {
            echo 'GAN training failed! Check logs.'
        }
    }
}