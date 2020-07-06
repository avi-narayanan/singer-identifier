const LEAD_KEY = 76;
const CHORUS_KEY = 67;
window.onload = function() {
    let input = document.getElementById("input");
    let sound = document.getElementById("sound");
    let p_lead = document.getElementById("lead");
    let p_chorus = document.getElementById("chorus");
    let lead_key_released = true;
    let chorus_key_released = true; 
    input.addEventListener("change", function(e){
        sound.src = URL.createObjectURL(this.files[0]);
        p_lead.textContent = "Lead: ";
        p_chorus.textContent = "Chorus: ";
        // not really needed in this exact case, but since it is really important in other cases,
        // don't forget to revoke the blobURI when you don't need it
        sound.onend = function(e) {
            URL.revokeObjectURL(this.src);
        }
    });

    // handle keydown
    document.addEventListener("keydown", event => {
        if (event.isComposing || event.keyCode === 229) {
        return;
        }
        if(event.keyCode === LEAD_KEY && lead_key_released){
            lead_key_released = false;
            p_lead.textContent += sound.currentTime.toFixed(2);
        }
        if(event.keyCode === CHORUS_KEY && chorus_key_released){
            chorus_key_released = false;
            p_chorus.textContent += sound.currentTime.toFixed(2);
        }
    });

    // handle keyup
    document.addEventListener("keyup", event => {
        if (event.isComposing || event.keyCode === 229) {
        return;
        }
        if(event.keyCode === LEAD_KEY){
            lead_key_released = true;
            p_lead.innerText += " - " + sound.currentTime.toFixed(2) + "; ";
        }
        if(event.keyCode === CHORUS_KEY){
            chorus_key_released = true;
            p_chorus.innerText += " - " + sound.currentTime.toFixed(2) + "; ";
        }
    });
}

