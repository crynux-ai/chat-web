if [ ! -f config/config.yml ]; then
    mkdir -p config
    cp config.yml.example config/config.yml
fi
python3 -m crynux_chat.main