#!/bin/bash

# Script to launch audio servers for music-making
case $1 in
	start )
		echo "Please select an audio device: (enter card number)"
		aplay -l | grep card
		read -p "> " hw
		echo "You selected HW:$hw"

		# Start JACK
		echo
		echo "Starting JACK..."
		pasuspender --\
			jackd -d alsa --device hw:$hw --rate 44100 --period 128 --softmode \
			&>/tmp/jackd.out &

		# Start fluidsynth
		echo
		echo "Starting Fluidsynth..."
		fluidsynth --server --no-shell --audio-driver=jack \
			--connect-jack-outputs --reverb=0 --gain=0.8 \
			/usr/share/sounds/sf2/FluidR3_GM.sf2 \
			&>/tmp/fluidsynth.out &
		sleep 1

		echo
		if pgrep jackd && pgrep fluidsynth
		then	echo "Audio servers running"
		else 
			echo "There was a problem starting the audio servers."
		fi

		;;
	stop )
		killall fluidsynth
		killall jackd
		echo "Thank you for the music."
		;;
	* )
		echo "Please specify start or stop..."
		;;
esac

