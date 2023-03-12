var jud = 0; 

function entersearch() {
    var event = window.event || arguments.callee.caller.arguments[0];
    if (event.keyCode == 13) {
        search();
    }
}

function search() {
    if (document.getElementById("wd").value != "") {
        window.location.href = "https://cn.bing.com/search?q=" + document.getElementById("wd").value;
        document.getElementById("wd").value = "";
    }
    return false;
}

function modechange() {
    if (jud == 0) {
        var obj = document.getElementById("mode");
        obj.setAttribute("href", "lib/css/dark.css");
        jud = 1;
    } else {
        var obj = document.getElementById("mode");
        obj.setAttribute("href", "lib/css/bright.css");
        jud = 0;
    }
}