FROM alpine:edge

RUN apk update

RUN apk add mosquitto

ADD mosquitto.conf /etc/mosquitto/mosquitto.conf

CMD ["/usr/sbin/mosquitto", "-v", "-c", "/etc/mosquitto/.config/mosquitto.conf"]