# build phase

FROM alpine AS build

RUN apk update && \
    apk add --no-cache \
    build-base

COPY src/ ./src/
COPY Makefile .

RUN make -B

# final image

FROM alpine

RUN apk update && \
    apk add --no-cache \
    libstdc++

COPY --from=build *.o ./app/
COPY --from=build fasttext ./app/

ENTRYPOINT [ "/bin/sh", "-c", "while true; do sleep 1000; done" ]