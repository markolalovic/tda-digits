size(200);
import graph3;


// compile with: asy -f png/pdf -noprc -render=8 torus.asy

pen surfPen=white;
pen arcPen=rgb(0, 0.38, 1);
currentprojection=perspective(5,4,4);
currentlight = light(0,0,100);

real R=2.5;
real a=1.2;

triple fs(pair t) {
  return ((R+a*Cos(t.y))*Cos(t.x),(R+a*Cos(t.y))*Sin(t.x),a*Sin(t.y));
}

surface s=surface(fs,(0,0),(360,360),8,8,Spline);
draw(s,surfPen,render(compression=Low,merge=true)); //+opacity(0.5)

pair p,q,v;
int i;
int j;
int n = 15;
int m = 2;
for(i=1;i<=n;++i) {
  int j = 0;
  p=(j*360/m,(i%n)*360/n);
  q=(((j+1)%m)*360/m,i*360/n);
  v=(((j+1/2)%m)*360/m,i*360/n);
  q=(j*360/m,((i%n)+1)*360/n);
  draw(fs(p)..fs((p+q)/2)..fs(q),arcPen); // vertical
  dot(fs(p));
}

m = 20;
i = 4;
for (j = 0; j < m; ++j) {
  p=(j*360/m,(i%n)*360/n);
  q=(((j+1)%m)*360/m,i*360/n);
  v=(((j+1/2)%m)*360/m,i*360/n);
  draw(fs(p)..fs(v)..fs(q),arcPen); // horizontal
  dot(fs(p));
}
