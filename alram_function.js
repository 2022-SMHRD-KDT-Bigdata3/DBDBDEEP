// oracledb 모듈을 불러옵니다.
const oracledb = require('oracledb');

// 알림 권한 요청 함수입니다.
function getNotificationPermission() {
  // 브라우저 지원 여부 체크
  if (!("Notification" in window)) {
    alert("데스크톱 알림을 지원하지 않는 브라우저입니다.");
  }
  // 데스크탑 알림 권한 요청
  Notification.requestPermission(function (result) {
    // 권한 거절
    if(result == 'denied') {
      alert('알림을 차단하셨습니다.\n브라우저의 사이트 설정에서 변경하실 수 있습니다.');
      return false;
    }
  });
}

// 주기적으로 데이터를 가져오는 함수입니다.
function getData() {
  setInterval(function() {
    oracledb.getConnection(
      {
        user: 'dbdb',
        password: 'dbdb',
        connectString: 'project-db-stu.ddns.net:1524/xe'
      },
      function(err, connection) {
        if (err) {
          console.error(err.message);
          return;
        }
        connection.execute(
          `SELECT records_object FROM records ORDER BY created_at DESC FETCH FIRST 1 ROWS ONLY`,
          function(err, result) {
            if (err) {
              console.error(err.message);
              return;
            }
            // records_object 값을 가져와서 알림으로 보여줍니다.
            if (result.rows.length > 0) {
              notify(result.rows[0][0]);
            }
          });
      });
  }, 30000);
}

// 알림을 띄우는 함수입니다.
function notify(msg) {
  var options = {
    body: msg
  }
  // 데스크탑 알림 요청
  var notification = new Notification("DBDBDeep", options);
  // 3초뒤 알람 닫기
  setTimeout(function(){
    notification.close();
  }, 3000);
}

// 알림 권한 요청 함수를 호출합니다.
getNotificationPermission();

// 주기적으로 데이터를 가져오는 함수를 호출합니다.
getData();
