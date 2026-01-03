Zero-Downtime Deployment System using Kubernetes
4
Project Overview
<img width="2862" height="1153" alt="Screenshot 2026-01-03 055357" src="https://github.com/user-attachments/assets/88f3113f-fe79-4939-8701-2be98ba01a2c" />

This project demonstrates how to deploy a web application without any service interruption using Kubernetes rolling updates.
The system ensures high availability by gradually replacing old application instances with new ones while keeping the service live for users.

Zero-downtime deployment is a critical DevOps practice used in production environments to avoid outages during application updates.

Objectives

Deploy applications with no downtime

Implement rolling update strategy

Ensure continuous service availability

Demonstrate real-world DevOps deployment practices

Understand Kubernetes Deployment & Service concepts

Tools & Technologies Used

Kubernetes – Container orchestration & rolling updates

Docker – Application containerization

Nginx – Web server for application hosting

GitHub – Version control & repository

Project Architecture

Application is containerized using Docker

Kubernetes Deployment manages multiple replicas

Kubernetes Service load-balances traffic

Rolling update replaces pods one-by-one

Users experience no downtime

Project Structure
zero-downtime-deployment/
│── app/
│   └── index.html
│── Dockerfile
│── deployment.yaml
│── service.yaml
│── README.md

 Application Versions

Version 1: Initial application deployment

Version 2: Updated application deployed without downtime

The change between versions is visible in the web page content.

Docker Configuration

Base image: Nginx

Application files copied to Nginx default directory

Lightweight and production-ready container

 Kubernetes Deployment Strategy

Replicas: 3 (ensures high availability)

Strategy: RollingUpdate

maxUnavailable: 1

maxSurge: 1

This configuration ensures:

At least 2 pods are always running

Traffic is never interrupted

 Kubernetes Service

Type: NodePort

Provides a stable endpoint for users

Automatically load-balances traffic across pods

Zero-Downtime Deployment Process

Deploy application version 1

Application runs with multiple replicas

Update Docker image to version 2

Kubernetes starts new pods

Old pods terminate gradually

Service remains available throughout

✅ No downtime experienced


git init
git status
git add .
git commit -m "Commit message"
git branch
git branch feature
git branch test
git branch bugfix
git branch experiment
git checkout branch_name
git merge branch_name
git remote add origin <repository-url>
git push -u origin main
git pull origin main
git clone <repository-url>
git log --oneline





Additionally, this project provided hands-on experience with Git Bash, GitHub, branching, merging, conflict resolution, and documentation best practices, making it a complete and industry-relevant implementation.
