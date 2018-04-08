var keys = [];
var channel_colours = _.range(16).map(i => 'hsl('+Math.round((i+7)%16*360/16)+',100%,50%)');

function MIDI_Message(data) {
    /*
    data is Uint8Array[3] with
    data[0] : command/channel
    data[1] : note
    data[2] : velocity
    */
    this.cmd = data[0] >> 4;
    this.channel = data[0] & 0xf; // 0-15
    this.type = data[0] & 0xf0;
    this.note = data[1];
    this.velocity = data[2];

    this.toString = function() {
        return 'type=' + this.type + 
            ' channel=' + this.channel + 
            ' note=' + this.note + 
            ' velocity=' + this.velocity;
    }
}

function write_to_console(text, color) {
    container = document.getElementById("console");
    // Create new paragraph element with text
    paragraph = document.createElement("p");
    var textnode = document.createTextNode(text);
    paragraph.appendChild(textnode);
    paragraph.className = "consoletext";
    paragraph.style.color = color;
    container.appendChild(paragraph);
    // Delete nodes if too many children
    if (document.getElementById('console').children.length > 200) {
        var firstchild = document.getElementById('console').children[0];
        document.getElementById('console').removeChild(firstchild);    
    }
    // Scroll to bottom of console container
    console_container = document.getElementById("console_container");
    console_container.scrollTop = console_container.scrollHeight;
}

function onMIDIMessage(data) {
    msg = new MIDI_Message(data.data);
    keys[msg.note].type = msg.type;
    keys[msg.note].channel = msg.channel;
    keys[msg.note].velocity = msg.velocity;
    write_to_console(msg.toString(), channel_colours[msg.channel]);
}

var p5sketch = function( p ) {
    var NUM_KEYS = 128;
    var NOTE_ON = 144;
    var NOTE_OFF = 128;

    function Key(index, key_w, key_h) {
        this.index = index;
        this.width = key_w - 4;
        this.height = key_h;
        this.left_edge = index * key_w;
        this.type = NOTE_OFF;
        this.channel = 0;
        this.velocity = 0;
        this.colour_off = p.color(0,0,10);
        this.colour_on = _.range(16).map(i => p.color(Math.round((i+7)%16*360/16),100,100,1) );

        this.draw = function() {
            // Always draw the empty key first, assuming note is off
            p.fill(this.colour_off);
            p.rect(this.left_edge, p.height-this.height, this.width, this.height);
            // Draw coloured key based on velocity (will end up transparent for NOTE_OFF since velocity=0)
            this.colour_on[this.channel]._array[3] = this.velocity / 125;
            // console.log(this.colour_on[this.channel]);
            p.fill(this.colour_on[this.channel]);
            p.rect(this.left_edge, p.height-this.height, this.width, this.height);
        }
    }

    p.setup = function() {
        p.createCanvas(p.windowWidth, p.windowHeight);
        p.noStroke();
        p.frameRate(30);
        p.colorMode(p.HSB); // Max values: 360, 100, 100, 1

        var keys_width = p.width / NUM_KEYS;
        var keys_height = 50;
        for (var i=0; i<NUM_KEYS; i++) {
            key = new Key(i, keys_width, keys_height)
            keys.push(key);
        }
    }

    p.draw = function() { 
        p.background(0);
        for (var i=0; i<NUM_KEYS; i++) {
            keys[i].draw();
        }
        p.fill(255);
    }

};
