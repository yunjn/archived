function matting() {
    var fileobj = $("#file0")[0].files[0];
    var form = new FormData();
    form.append("file0", fileobj);
    $.ajax({
        type: 'POST',
        url: "matting",
        data: form,
        async: false,       //同步执行
        processData: false, // 告诉jquery要传输data对象
        contentType: false, //告诉jquery不需要增加请求头对于contentType的设置
        success: function (arg) {
            console.log(arg)
            out = arg.result;
        }, error: function () {
            console.log("后台处理错误");
        }
    });
    $("#img1").attr("src", "data:;base64," + out);
}

function change_bg() {
    var fileobj = $("#file1")[0].files[0];
    var form = new FormData();
    form.append("file1", fileobj);
    $.ajax({
        type: 'POST',
        url: "change_bg",
        data: form,
        async: false,       //同步执行
        processData: false, // 告诉jquery要传输data对象
        contentType: false, //告诉jquery不需要增加请求头对于contentType的设置
        success: function (arg) {
            console.log(arg)
            out = arg.result;
        }, error: function () {
            console.log("后台处理错误");
        }
    });
    $("#img3").attr("src", "data:;base64," + out);
}