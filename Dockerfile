FROM goatcommunity/data-preparation-base:0.04

# Install GTFSTIDY 
# RUN bash -c "cd /home && wget https://go.dev/dl/go1.19.2.linux-amd64.tar.gz && rm -rf /usr/local/go && tar -C /usr/local -xzf go1.19.2.linux-amd64.tar.gz"
# ENV PATH="${PATH}:/usr/local/go/bin" 
# ENV GOPATH="/root/go"
# ENV PATH="${GOPATH}/bin:${PATH}"
# RUN bash -c "go install github.com/patrickbr/gtfstidy@latest"

# Copy poetry.lock* in case it doesn't exist in the repo
COPY ./pyproject.toml ./poetry.lock* /app/
ENV PYTHONPATH "${PYTHONPATH}:."
WORKDIR /app/

# Install python dependencies
RUN bash -c "poetry install --no-root"

# Install aws cli
WORKDIR /tmp/
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

# Install s3fs
RUN apt update
RUN apt install -y s3fs

# Install kart 
RUN apt-get install -y libtinfo5
RUN wget "https://goat-db-schemas.s3.eu-central-1.amazonaws.com/kart.deb"
RUN dpkg -i kart.deb

# install posgresql-client
RUN sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
RUN apt-get update
RUN apt-get install -y postgresql-client-15

# install nano and make default git editor
RUN apt install -y nano
RUN git config --global core.editor "nano"

# Install zip
RUN apt install -y zip

# Install Java 11
RUN echo 'deb http://deb.debian.org/debian bullseye main' > /etc/apt/sources.list
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get clean;



# # get 3citydb importer 
# RUN wget "https://github.com/3dcitydb/importer-exporter/releases/download/v5.3.0/3DCityDB-Importer-Exporter-5.3.0.zip"
# RUN unzip 3DCityDB-Importer-Exporter-5.3.0.zip
# RUN rm 3DCityDB-Importer-Exporter-5.3.0.zip
# RUN mv 3DCityDB-Importer-Exporter-5.3.0 /opt/3dcitydb
# RUN chmod +x /opt/3dcitydb/bin/impexp

WORKDIR /app/

ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]