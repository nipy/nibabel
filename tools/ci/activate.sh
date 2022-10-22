if [ -e venv/bin/activate ]; then
    source venv/bin/activate
elif [ -e venv/Scripts/activate ]; then
    source venv/Scripts/activate
else
    echo Cannot activate virtual environment
    ls -R venv
    false
fi

