# build phase

FROM alpine AS build

RUN apk update && \
    apk add --no-cache \
    build-base \
    fasttext-dev

COPY src/ ./src/
COPY Makefile .

RUN make -B

# final image

FROM alpine

RUN apk update && \
    apk add --no-cache \
    libstdc++ \
    fasttext-libs

COPY --from=build *.o ./app/
COPY --from=build fasttext ./app/

ENTRYPOINT [ "tail", "-f", "/dev/null" ]