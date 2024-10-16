# os/gpu
FROM ubuntu:22.04

# packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    build-essential \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    zlib1g-dev \
    wget \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl \
    ca-certificates \
    tmux \
    locales \
    && apt-get clean

# locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


# user
ARG USER_NAME=container_user
ARG USER_ID=2000
ARG GROUP_ID=2000
ARG GROUP_NAME=worker

RUN groupadd -g ${GROUP_ID} ${GROUP_NAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME} && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /workspace
RUN chown -R 0:${GROUP_ID} /workspace && \
    chmod -R 775 /workspace

USER ${USER_NAME}


# pyenv
ENV PYENV_ROOT=/workspace/.pyenv
ENV PATH="$PYENV_ROOT/bin:$PATH"
ARG PYTHON_VERSION=3.11
ENV PYTHON_VERSION ${PYTHON_VERSION}
RUN curl https://pyenv.run | bash
RUN echo 'export PYENV_ROOT="$PYENV_ROOT"' >> ~/.bashrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc"
RUN /bin/bash -c "pyenv install ${PYTHON_VERSION} && pyenv global ${PYTHON_VERSION}"

# Poetry
ENV POETRY_HOME=/workspace/poetry
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3

