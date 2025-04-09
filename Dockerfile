# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install openssh-server and sudo
RUN apt-get update && apt-get install -y openssh-server sudo

# Create SSH run directory
RUN mkdir /var/run/sshd

# Create a new user "devuser" with password "password" (adjust as needed)
RUN useradd -ms /bin/bash devuser && echo "dockfabri:dockerpassword93" | chpasswd && adduser devuser sudo

# Expose port 22 for SSH access
EXPOSE 22

# Start SSH daemon when the container runs
CMD ["/usr/sbin/sshd", "-D"]
