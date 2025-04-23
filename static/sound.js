var synth = window.speechSynthesis;

// Inisialisasi array suara yang tersedia
var available_voices = [];

// Variabel untuk menampung suara yang dipilih
var selected_voice = '';

// Fungsi untuk memperbarui daftar suara yang tersedia setelah dimuat
function populateVoices() {
  available_voices = synth.getVoices(); // Mendapatkan daftar suara yang tersedia di browser

  // Memeriksa apakah suara tersedia
  if (available_voices.length === 0) {
    console.log("Tidak ada suara yang ditemukan.");
    return;
  }

  // Mencari suara berdasarkan bahasa Indonesia (id-ID)
  for (var i = 0; i < available_voices.length; i++) {
    if (available_voices[i].lang === 'id-ID') {
      selected_voice = available_voices[i];
      break;
    }
  }

  // Jika tidak ada suara Bahasa Indonesia, pilih suara pertama yang tersedia
  if (!selected_voice) {
    selected_voice = available_voices[0];
  }

  console.log("Suara yang dipilih: ", selected_voice);
}

// Memeriksa apakah browser mendukung event 'onvoiceschanged' dan menambahkan event listener
if (synth.onvoiceschanged !== undefined) {
  synth.onvoiceschanged = populateVoices;
} else {
  populateVoices(); // Jika browser tidak mendukung 'onvoiceschanged'
}
   

var $messages = $('.messages-content'),
    d, h, m,
    i = 0;

$(window).load(function() {
  $messages.mCustomScrollbar();
  setTimeout(function() {
    $('<div class="message new"><figure class="avatar"><img src="/static/robo1.jpg" /></figure>' + 'I am your chatbot. Based on your symptoms here i predict your disease. And i also suggest analgesics,treatment scans,diet for that prediction disease.' + '</div>').appendTo($('.mCSB_container')).addClass('new');
        setDate();
        updateScrollbar();
        textToSpeech('I am your chatbot. Based on your symptoms here i predict your disease. And i also suggest analgesics,treatment scans,diet for that prediction disease.' );
  }, 100);
});

function updateScrollbar() {
  $messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {
    scrollInertia: 10,
    timeout: 0
  });
}

function setDate(){
  d = new Date()
  if (m != d.getMinutes()) {
    m = d.getMinutes();
    $('<div class="timestamp">' + d.getHours() + ':' + m + '</div>').appendTo($('.message:last'));
  }
}

function insertMessage() {
  msg = $('.message-input').val();
  if ($.trim(msg) == '') {
    return false;
  }
  $('<div class="message message-personal">' + msg + '</div>').appendTo($('.mCSB_container')).addClass('new');
  setDate();
  $('.message-input').val(null);
  updateScrollbar();
  setTimeout(function() {
    $('<div class="message loading new"><figure class="avatar"><img src="/static/robo1.jpg" /></figure><span></span></div>').appendTo($('.mCSB_container'));
    updateScrollbar();
    

    fetch(`${window.origin}/entry`, {
      method: "POST",
      credentials: "include",
      body: JSON.stringify(msg),
      cache: "no-cache",
      headers: new Headers({
        "content-type": "application/json"
      })
    })
    .then(function(response) {
      if (response.status !== 200) {
        console.log(`Looks like there was a problem. Status code: ${response.status}`);
        return;
      }
      response.json().then(function(data) {
        console.log(data);

        $('.message.loading').remove();
        $('<div class="message new"><figure class="avatar"><img src="/static/robo1.jpg" /> </figure>' + data.name  + '</div>').appendTo($('.mCSB_container')).addClass('new');
        setDate();
        updateScrollbar();
        textToSpeech(data.name);
       
   
      });
    })
    .catch(function(error) {
      console.log("Fetch error: " + error);
   });
   
  }, 1000 + (Math.random() * 20) * 100);
}

$('.message-submit').click(function() {
  insertMessage();
});

$(window).on('keydown', function(e) {
  if (e.which == 13) {
    insertMessage();
    return false;
  }
})

var Fake = [
  'Hi im your chatbot ',
  'please enter your name ',
  'Please Enter Your age',
  'good.....What is your comfortable level for investment loss (in %) <input type="range" value="50" min="0" max="100" step="10" />',
  'we are Predicting... <div class="loading-img"><img src="5.png"  alt=""/></div>',
  'great.. do you want to predict another? <button class="buttonx sound-on-click">Yes</button> <button class="buttony sound-on-click">No</button> ',
  'Bye',
  ':)'
]

function fakeMessage() {
  msg = $('.message-input').val()
  if (msg != '') {
    return false;
  }
  $('<div class="message loading new"><figure class="avatar"><img src="/static/robo1.jpg" /></figure><span></span></div>').appendTo($('.mCSB_container'));
  updateScrollbar();

  setTimeout(function() {
    $('.message.loading').remove();
    fetch(`${window.origin}/entry`, {
      method: "POST",
      credentials: "include",
      body: JSON.stringify(msg),
      cache: "no-cache",
      headers: new Headers({
        "content-type": "application/json"
      })
    })
    .then(function(response) {
      if (response.status !== 200) {
        console.log(`Looks like there was a problem. Status code: ${response.status}`);
        return;
      }
      response.json().then(function(data) {
        console.log(data);
        $('<div class="message new"><figure class="avatar"><img src="/static/robo1.jpg" /></figure>' + data.name + '</div>').appendTo($('.mCSB_container')).addClass('new');
        setDate();
        updateScrollbar();
    
   
      });
    })
    .catch(function(error) {
      console.log("Fetch error: " + error);
   });
   
    i++;
  }, 1000 + (Math.random() * 20) * 100);

}






                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  


