import time
import functools

import csv
import fire
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from torch.profiler import profile, record_function, ProfilerActivity
# from vllm import LLM, SamplingParams

# model = LLM(model="TheBloke/samantha-mistral-instruct-7B-AWQ", quantization="awq", dtype="half")

lorem_ipsum = """

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam tincidunt finibus justo, vitae dignissim erat ultrices et. Praesent eget mattis velit, in finibus purus. Nam et mauris ultrices, suscipit sem et, rutrum enim. Maecenas vulputate ligula non est venenatis feugiat. Vestibulum tristique, nulla id vulputate pulvinar, mauris enim suscipit nisl, imperdiet porta ipsum turpis molestie massa. Nam sagittis lacus a risus dapibus, id molestie justo dapibus. Nullam egestas enim volutpat massa fermentum laoreet. Donec eleifend urna vel augue vehicula sollicitudin. Etiam vitae lacus ornare, blandit eros non, laoreet sem. Curabitur sollicitudin sodales libero eu venenatis. Nam eget lorem libero. In quis enim a nulla malesuada sollicitudin ut sed magna. Nulla aliquet risus id lacus porttitor, in gravida quam cursus. Duis lacinia metus eget lacus pellentesque porta. Aliquam vehicula aliquet pulvinar.

Sed sagittis vitae ante sed consequat. Duis faucibus justo sit amet magna consequat, sit amet pharetra elit porttitor. Integer convallis laoreet justo, vel consectetur eros rutrum in. Sed eleifend libero nec ipsum mattis, in cursus tortor interdum. Nunc vel odio a est semper iaculis eu eget neque. In hac habitasse platea dictumst. Vestibulum condimentum sapien nec est sollicitudin cursus. Vivamus aliquam imperdiet erat quis molestie. Nulla facilisi. Morbi vel tortor erat. Vivamus ac ligula lectus. Vestibulum et mauris molestie, scelerisque elit id, placerat est. Sed vestibulum pretium tempor. Aenean eget congue metus. Morbi efficitur sollicitudin vestibulum. Fusce mattis faucibus arcu ut egestas.

Aliquam posuere interdum ex in aliquet. Sed quis ullamcorper sem, sed porttitor risus. Nulla facilisi. Nulla vel venenatis tortor. Aenean dapibus feugiat ipsum, a varius nunc molestie ac. Vestibulum congue molestie elementum. Suspendisse ultricies condimentum magna. Nunc accumsan sit amet enim ultricies tempus. Maecenas convallis tellus non sapien mollis aliquam. Nam pellentesque malesuada tempor. Duis sagittis consectetur sapien at commodo. Suspendisse eget turpis a sem dapibus lobortis.

Aliquam erat volutpat. Duis nec viverra nunc. Vestibulum blandit ante metus, et vulputate lorem commodo quis. Donec vitae sollicitudin tortor. Integer auctor scelerisque nulla, sit amet consequat nisl lobortis eu. Proin non leo magna. Vivamus ut diam in nulla posuere pharetra. Etiam ante justo, vestibulum vitae porta nec, euismod id odio. Nullam in tristique elit, in interdum magna. Nulla est leo, eleifend ac diam sed, cursus ullamcorper ipsum. Quisque a dolor quis risus ultrices finibus. Pellentesque eu lacus quis elit rhoncus porta. Etiam semper imperdiet nisi id fermentum.

Aliquam fermentum, leo in tincidunt accumsan, ante erat vulputate orci, vitae molestie eros turpis at felis. In hac habitasse platea dictumst. In eleifend placerat augue ac ullamcorper. Duis convallis laoreet tempor. Sed feugiat efficitur blandit. Praesent venenatis metus eget molestie congue. Curabitur sit amet imperdiet nulla. Maecenas luctus risus ac tincidunt accumsan. Nullam elementum ante eu eros porta, at varius enim bibendum. Nam at rutrum massa. Maecenas consequat sed metus vitae pretium. Maecenas ullamcorper venenatis euismod. Quisque non diam a mi accumsan porttitor. Aliquam vitae auctor metus.

Quisque eget efficitur erat. Suspendisse potenti. In fermentum viverra vulputate. Fusce facilisis orci dui, ac imperdiet sem vehicula vitae. Donec ut lectus sed mi suscipit dapibus. Pellentesque ut faucibus massa. Duis lacus nibh, finibus nec aliquet nec, eleifend vel tortor. Suspendisse id consectetur odio. Suspendisse imperdiet gravida convallis. In et ultrices velit. Nunc lacus orci, bibendum eu venenatis vitae, sodales quis ante.

In et pharetra est. Sed tincidunt erat et velit faucibus porttitor. Aenean eu accumsan nisl, sed aliquam orci. Praesent nec posuere eros, et consequat libero. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Praesent id libero felis. Mauris tincidunt lectus ut nisi porttitor, non tempus ante vestibulum. Proin metus ligula, sollicitudin sit amet arcu sodales, pulvinar rutrum elit. Nullam eleifend lorem nec placerat varius. Nullam id elit ac libero molestie tincidunt id egestas neque. Donec luctus consequat tristique. In ornare auctor neque ac maximus. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Duis a venenatis magna. Integer sit amet euismod magna. Vivamus lobortis quam vitae egestas consectetur.

Duis ornare justo ac nunc elementum pretium. Donec ante quam, placerat non risus fringilla, facilisis hendrerit ligula. Maecenas mollis mi quis pharetra dignissim. Integer elementum, mi posuere tempus sodales, ipsum dui tristique lectus, vitae viverra nisl lacus sed lorem. Aliquam erat volutpat. In dictum egestas pharetra. Sed porta lacinia neque accumsan rutrum. Ut ligula felis, venenatis nec molestie sit amet, ultrices maximus magna. In pretium in diam et efficitur. Suspendisse ante felis, molestie et pulvinar vitae, dignissim ac velit.

Praesent bibendum placerat erat, sed pellentesque justo egestas vitae. Etiam iaculis felis a urna suscipit, eleifend convallis est feugiat. Praesent quis finibus diam, eu laoreet orci. Curabitur ac ultrices tellus, ullamcorper mattis tellus. Sed auctor tortor vitae nunc viverra, quis consectetur odio cursus. Pellentesque ultricies, libero quis facilisis ultricies, lorem est lacinia quam, eget varius felis ex a mauris. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aliquam convallis urna eget mauris tempor placerat eget nec nisl. Mauris ac risus semper, pellentesque ante at, laoreet tellus. Nullam luctus bibendum nisi ac dapibus. Vivamus porta tellus erat, at finibus nulla semper vel.

Duis aliquam pharetra massa, et sagittis ligula porta at. Nullam facilisis erat at rutrum malesuada. Morbi posuere eros a augue tincidunt, eu consequat risus condimentum. Vivamus dolor est, rutrum quis bibendum ut, scelerisque a nisi. Donec efficitur aliquam eros, eget efficitur velit laoreet ut. Nulla facilisi. Sed eget est nec enim elementum vestibulum sed vitae tortor.

Ut nisl augue, tincidunt ut maximus ut, ultricies sit amet turpis. Vivamus faucibus, metus ac dapibus eleifend, massa arcu porta tellus, nec convallis lorem quam sit amet tortor. Quisque sit amet ligula at urna finibus blandit. Proin sed ullamcorper dolor. Ut nec aliquet urna, pellentesque rutrum purus. Duis nulla erat, malesuada sit amet risus vel, aliquet ullamcorper ex. Nullam aliquam risus nulla, quis maximus nulla tempor in. Phasellus vitae purus sit amet magna varius luctus id non leo. Sed mattis dolor diam, et rhoncus ante tempor at. Nunc fermentum ullamcorper dui, vitae ornare elit porttitor feugiat.

Cras eu est vel sem venenatis volutpat id nec eros. Ut id est mauris. Donec convallis justo quam, id sagittis ante sollicitudin id. Mauris nec vestibulum lectus. Integer id diam in nisl consectetur consequat. Maecenas placerat ex elit, vel vestibulum augue consequat et. In non iaculis nisi. Nulla facilisi. Vivamus ac scelerisque augue. Donec eget lacus a dolor ornare elementum ac a purus. Pellentesque sodales risus vitae malesuada pellentesque. Curabitur eget rutrum turpis. Aliquam diam augue, pharetra vel convallis nec, lobortis non justo. Curabitur aliquam dignissim augue quis ullamcorper. In tempor lorem eu elit iaculis rhoncus.

Vestibulum dapibus nulla eget dolor tincidunt vestibulum. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Nunc viverra metus ac ligula pharetra semper. Integer eget ultrices sapien, sed laoreet leo. In nec tortor non diam lobortis fringilla at vitae odio. Morbi velit nunc, pulvinar et lacinia blandit, ullamcorper vel felis. Integer accumsan pulvinar sem et condimentum. Nunc non tortor risus. In mattis vehicula nulla non suscipit. Nulla sed pulvinar purus. Proin in tempor augue, quis dictum justo.

Aliquam eleifend massa ac est commodo molestie. Nulla eros sem, viverra sit amet erat id, varius dapibus velit. Praesent imperdiet mattis erat nec hendrerit. Integer laoreet libero nec orci ultricies volutpat. Duis ut nulla euismod, pellentesque nulla sit amet, rhoncus nulla. Quisque aliquam feugiat turpis at maximus. Morbi rhoncus urna eu sapien tristique commodo. Donec mollis lorem ac luctus sollicitudin. Donec nisi lacus, fermentum consectetur dictum blandit, ornare non nibh. Aliquam diam dui, lobortis ut efficitur finibus, tincidunt nec metus. Maecenas posuere consectetur erat, vitae ornare ante sodales eget. Vestibulum eu fringilla mi. Ut euismod nulla non ligula convallis, nec condimentum lorem sagittis. Aliquam pharetra mollis pulvinar. In condimentum eros felis, in facilisis magna tempor quis. Duis molestie a eros a convallis.

Suspendisse eget eleifend augue, sed commodo nisl. Integer vel augue tellus. In ligula orci, porta id ligula mollis, viverra eleifend lorem. Mauris rutrum ligula id nisi congue posuere. Proin eros purus, iaculis nec maximus elementum, rutrum ut nunc. Mauris blandit felis pretium ipsum porttitor, a rutrum risus commodo. Maecenas nisl elit, dignissim eu augue eu, fringilla tincidunt neque. Mauris iaculis feugiat lectus, eget bibendum ex laoreet sit amet. Suspendisse nec ex sed lacus accumsan molestie. Proin tempor nulla neque, at lacinia tortor gravida ut. Aenean eget ullamcorper massa.

Morbi sed ex lacus. Mauris sed nisl mauris. Maecenas consequat vulputate odio sit amet hendrerit. In sem libero, efficitur at purus eu, pulvinar dignissim dolor. In facilisis convallis felis. Aliquam sit amet enim nisi. Aenean a quam ut ipsum tincidunt imperdiet id ac nisi. Ut ut consectetur nibh, sed ultrices magna. Nunc in velit nunc. Cras quis ornare purus. Praesent rutrum, urna vitae mattis pellentesque, lorem metus mollis arcu, sit amet consequat tellus eros non felis. Mauris vel sem quis erat aliquam sodales commodo ac nisl. Quisque ut erat ipsum.

Ut sit amet mauris venenatis, faucibus nunc in, iaculis est. Nullam in laoreet lectus. Suspendisse faucibus magna a enim elementum vulputate. Etiam interdum libero ex, finibus facilisis leo aliquam id. Nam sit amet volutpat nunc, ac viverra dui. Morbi nec nulla elit. Nullam sapien turpis, volutpat eu iaculis sit amet, dictum nec justo. Integer nec euismod lectus. Curabitur sed malesuada purus, in viverra odio. Pellentesque augue est, hendrerit vitae tempor sit amet, cursus in dolor. Phasellus non ullamcorper purus. Mauris ultricies nibh a justo porttitor tempus.

Nam posuere metus quam, sit amet auctor ligula porta id. Aliquam ullamcorper urna at eros convallis volutpat. Donec at dolor cursus, sagittis nibh sed, cursus ligula. Donec enim est, sollicitudin pellentesque sollicitudin id, blandit eu elit. Ut hendrerit sollicitudin arcu, ut placerat sem luctus non. Morbi consequat bibendum tellus eu rhoncus. Donec turpis libero, scelerisque vel venenatis eu, laoreet nec magna. In hac habitasse platea dictumst. Phasellus vestibulum velit a porta facilisis. Cras eu blandit diam. Sed iaculis lacus nibh, et mattis odio faucibus sed. In hac habitasse platea dictumst. In hac habitasse platea dictumst. Praesent eleifend fringilla sem. Fusce dapibus massa non pretium efficitur.

Nam interdum volutpat neque, facilisis pharetra magna finibus at. Ut id vestibulum enim. Curabitur tempus non massa sit amet auctor. Praesent et ante quis urna egestas porta et id quam. Suspendisse potenti. In maximus lacus a lectus tempus, sit amet maximus diam efficitur. Vestibulum eleifend, ex sit amet fringilla blandit, magna purus semper nisi, vel pellentesque turpis mi quis ipsum. Maecenas sed justo libero. In id condimentum erat, quis vulputate eros. Pellentesque convallis justo augue, id efficitur ex hendrerit non. Nulla bibendum purus a elit mollis, eget tristique elit pellentesque.

Curabitur tristique nulla dolor, eget dapibus sem tincidunt eu. Suspendisse aliquam posuere dolor at tincidunt. Donec eget placerat diam. Nullam eleifend sapien ornare augue pretium, et bibendum orci aliquet. Suspendisse sollicitudin urna turpis, at scelerisque arcu finibus id. In porta at magna et elementum. Vestibulum mollis faucibus tortor, consectetur accumsan felis vehicula a. Vivamus vitae posuere libero. Morbi auctor, nulla sed pretium finibus, sem erat vehicula augue, eu elementum tortor quam nec eros. Praesent venenatis sollicitudin arcu, non posuere turpis mollis ac. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos.

Praesent convallis leo at velit tincidunt, ultricies volutpat leo lobortis. Phasellus lorem augue, consequat eget lobortis a, egestas nec lectus. Morbi pulvinar justo leo, ut lacinia lorem malesuada id. Mauris scelerisque porttitor aliquet. Maecenas a metus et erat semper imperdiet. Morbi pulvinar purus nunc, sit amet bibendum ipsum hendrerit quis. Donec ultrices augue eros, ut rutrum neque rhoncus ut. Pellentesque augue erat, hendrerit tincidunt ipsum ut, placerat varius orci. Vestibulum commodo arcu libero, vel dictum felis vehicula vitae. Donec felis magna, efficitur vel tortor vitae, blandit egestas odio. Pellentesque mattis felis ac porttitor lacinia. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam non nisl aliquam, gravida ante vel, semper risus. Integer efficitur dapibus magna, eu condimentum tellus aliquet in. Duis fermentum neque non mi fermentum, sed faucibus sapien egestas. Praesent luctus maximus velit nec porta.

Aenean metus tellus, condimentum mollis congue non, viverra id tellus. Duis ullamcorper nibh in felis porta, quis dignissim arcu accumsan. Donec tempus finibus sapien, quis mollis nisl fringilla ullamcorper. Nam faucibus, nulla ut iaculis consectetur, dui arcu vulputate lacus, ac dignissim neque arcu ac ligula. Vivamus et convallis elit. Nunc suscipit est ante. Aenean eget ipsum at tortor tempus tempor. Fusce sodales, felis sed consectetur consectetur, sem lectus vestibulum nulla, non porta diam eros ac magna. Sed cursus neque placerat eros convallis, a tristique diam convallis. Sed efficitur volutpat blandit. Vestibulum pharetra nulla vitae felis vulputate vulputate.

Integer ipsum lectus, imperdiet et rhoncus in, congue iaculis lacus. Pellentesque porta, est vel volutpat rutrum, leo nisi sagittis tellus, et cursus ligula nunc vitae odio. Aliquam malesuada justo at neque blandit blandit. Vivamus ut lectus quis velit commodo aliquam. Donec egestas scelerisque semper. Proin venenatis pharetra ex vitae euismod. Aliquam turpis ipsum, venenatis at efficitur vitae, mattis sit amet sem. Suspendisse et tempor libero, fermentum suscipit justo. Phasellus dictum nibh arcu. Curabitur vitae eleifend purus, sed interdum velit. In iaculis vitae massa et ullamcorper. Sed a placerat lectus, ut elementum nunc. Nam ut sem dictum, maximus lectus ut, sodales orci. Donec ultricies condimentum tellus sed volutpat. Maecenas consectetur tempor lorem, eu vehicula lacus dapibus varius.

Maecenas sit amet sapien nec odio dignissim pretium. Cras libero ligula, scelerisque venenatis ullamcorper vel, tincidunt hendrerit urna. Quisque a neque tincidunt, consequat dolor vel, luctus libero. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Donec ac metus et odio venenatis varius. Mauris quis sapien in sem rutrum hendrerit dignissim eu libero. Mauris eu ligula sapien. Integer a dui scelerisque, feugiat magna sit amet, iaculis nisl. Ut et risus id enim sollicitudin consequat in sit amet urna. Etiam egestas euismod quam et condimentum. Praesent varius dignissim risus, id interdum tellus. Quisque lobortis erat eget ligula tristique scelerisque. Suspendisse potenti. Duis in justo massa. Integer eu sagittis quam.

Sed vulputate urna massa, et vehicula tortor hendrerit vel. Cras tincidunt tristique bibendum. Donec vel consectetur tortor. Duis blandit sem et massa aliquam hendrerit. In ac libero condimentum, molestie neque ac, semper odio. Nunc lacinia magna a maximus porta. Nam purus lorem, suscipit ac urna ac, blandit lacinia ante. In hac habitasse platea dictumst. Nulla molestie justo non aliquet venenatis. Ut vehicula fringilla pulvinar. Phasellus sit amet odio in felis lobortis ultricies aliquet pellentesque augue. Sed auctor mauris at nibh suscipit, sed venenatis ante sollicitudin. Sed at convallis enim, sit amet rhoncus risus. Curabitur volutpat at diam et aliquam.

Quisque elementum volutpat ligula ac aliquam. Sed a sapien et enim maximus aliquam. Praesent luctus volutpat sem, malesuada ultricies orci fermentum sed. Aenean ex libero, fringilla id felis vitae, tincidunt rhoncus mi. Proin placerat ut sem at commodo. Nam vestibulum venenatis luctus. Sed pharetra lorem quis justo varius, sit amet pellentesque dolor pretium. Ut viverra leo tellus. Nunc aliquam a nulla at volutpat. Ut non augue rutrum, vehicula arcu vitae, imperdiet urna. Proin commodo purus eget mi dapibus scelerisque. Morbi eget imperdiet turpis. Mauris sem sem, sodales sed eleifend consectetur, elementum id velit. Aliquam sed mauris ut dui blandit accumsan. Ut metus mauris, varius eu congue a, congue sed velit.

Nullam quis elit eget arcu tincidunt accumsan. Suspendisse nec commodo neque, quis placerat nibh. Donec bibendum faucibus enim, ac tempor tortor imperdiet eu. Suspendisse risus erat, fermentum eget tincidunt at, condimentum sed enim. Maecenas vel tempor nibh. Vivamus vestibulum varius nulla, nec malesuada magna faucibus vitae. Phasellus ac odio euismod, porttitor ipsum ut, tempus massa. Praesent bibendum orci et felis volutpat pellentesque. Maecenas hendrerit euismod nulla vel lacinia. Donec consectetur condimentum ligula eu malesuada. Morbi vel porta tellus. Nulla tincidunt odio nec dui imperdiet vulputate vel accumsan tortor. Duis tempus interdum ullamcorper. Nunc ut mi congue, convallis magna sit amet, viverra justo.

Pellentesque at ultricies massa. Proin dolor est, maximus at nibh eu, lobortis malesuada elit. Aenean venenatis at eros eu facilisis. Curabitur eu sem eget odio pellentesque posuere. Ut luctus neque velit, congue suscipit odio elementum vitae. Curabitur eros ipsum, convallis ac vestibulum in, convallis malesuada purus. Donec scelerisque consequat eros eget facilisis. Vivamus condimentum congue ligula mollis condimentum. Nulla luctus justo feugiat tortor fringilla tincidunt.

Etiam eu purus sapien. Lorem ipsum dolor sit amet, consectetur adipiscing elit. In maximus sollicitudin arcu, vestibulum lacinia neque feugiat ut. Morbi eu leo eu urna bibendum tincidunt. Vivamus id tellus sit amet risus venenatis venenatis in quis arcu. In eu aliquam ex. Duis vehicula erat vel scelerisque tempor. Nulla at imperdiet ipsum. Nullam lacinia ipsum et egestas porta.

Nulla sed nulla ullamcorper, sagittis ipsum eget, porta magna. Phasellus aliquet neque a venenatis dignissim. In aliquet leo quis ipsum elementum, nec pharetra sapien rutrum. Maecenas tincidunt euismod quam a sodales. Donec erat sapien, bibendum accumsan dui sed, vulputate tempor purus. Donec sed gravida magna. Aliquam vehicula sollicitudin nulla at consectetur. Etiam nec leo dictum, ullamcorper odio at, pharetra nulla.

Curabitur at nulla sem. Curabitur in interdum quam. In id mattis sem. Vivamus ullamcorper bibendum ultricies. Morbi semper nulla ut mollis ultrices. Proin non iaculis augue. Suspendisse potenti. Donec non dui nibh. Fusce feugiat, libero a congue semper, leo velit sollicitudin ligula, et blandit orci massa id lacus. Curabitur vel gravida tellus. Duis id libero id quam mollis feugiat. Proin molestie lectus id neque rhoncus elementum. Interdum et malesuada fames ac ante ipsum primis in faucibus.

In maximus pulvinar nisl. Cras leo risus, iaculis id ipsum sed, blandit faucibus leo. Aliquam in turpis vitae urna faucibus euismod. Ut posuere dictum ipsum a gravida. Fusce faucibus turpis nec sapien iaculis interdum. Nulla facilisi. Fusce ligula risus, finibus a ornare sed, pellentesque at mauris. Aliquam nec tempus augue. Quisque ut augue nec risus tempor posuere in eget lorem. Donec finibus neque in cursus cursus. Etiam ex mi, placerat a purus finibus, auctor commodo arcu.

Phasellus ut tincidunt enim, sed mollis lacus. Quisque nec dolor a ligula malesuada euismod non vel nulla. Nunc et velit at dui hendrerit semper placerat id lorem. Nullam viverra facilisis rutrum. Maecenas nec facilisis ante. Morbi quis lacus ac nisl dapibus facilisis. Ut ut erat dictum mi feugiat viverra ultrices non nibh. Phasellus laoreet mauris id placerat ullamcorper. Sed dictum elit eu tellus lobortis tempor. Suspendisse pretium enim sem, non ultricies enim vehicula eget. Proin interdum laoreet viverra. Vestibulum suscipit est felis, id accumsan nisl congue vel.

Nulla neque nisl, tempus sit amet cursus at, malesuada a purus. Proin sit amet enim pellentesque, gravida tellus quis, lacinia ligula. Pellentesque tempus pulvinar tellus eu finibus. Quisque dignissim pretium tellus non ornare. Nulla malesuada sapien egestas odio maximus, ac accumsan leo tempor. Morbi maximus lacus lectus, sit amet lobortis quam ultricies vel. Vestibulum vitae sollicitudin erat. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Mauris at vulputate nibh. Curabitur et ipsum vulputate, consectetur neque ac, placerat erat. Vestibulum a velit turpis.

Curabitur malesuada, felis eu convallis fringilla, justo odio lobortis justo, pharetra pharetra lectus turpis eu ante. Donec a nunc et lectus malesuada varius. Quisque venenatis sagittis turpis et mattis. Cras eget ipsum nec nulla facilisis venenatis vitae quis libero. Donec nec euismod lacus. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Nam sit amet consectetur felis. Sed orci odio, malesuada sit amet finibus ac, gravida in massa.

In blandit egestas dolor, sed commodo libero volutpat sed. Quisque interdum odio non eros consectetur malesuada. Mauris commodo eros diam, a lacinia nulla tempus ut. Suspendisse rutrum felis eu mi convallis, id congue lectus suscipit. Vestibulum aliquet vulputate eleifend. Duis dolor magna, blandit in rutrum sit amet, volutpat sed tellus. Nam quis turpis vel augue finibus porttitor quis vitae neque. Morbi et ligula lacinia, facilisis tortor sit amet, varius dolor. Vestibulum rutrum lorem eget tincidunt porttitor. Cras vel porta eros. Mauris vestibulum diam in libero faucibus, et mattis arcu hendrerit. Donec ac luctus eros. Duis convallis ipsum euismod diam pharetra efficitur. Donec elementum mauris in feugiat porta.

Quisque volutpat elementum lorem nec condimentum. Nam pellentesque velit ornare, eleifend ligula sodales, elementum nibh. Integer condimentum lobortis nunc, ut semper ex sodales quis. Nunc tincidunt malesuada dui, vel volutpat arcu tincidunt ac. Nam facilisis tempus lorem, vitae pulvinar leo dapibus a. Praesent pharetra turpis purus, aliquam pellentesque turpis consequat at. Nam ac nunc diam. Sed sed laoreet justo.

Donec ut justo mauris. Mauris vestibulum ultricies tempor. Nunc tincidunt lorem tortor, nec porta sem dignissim quis. In hac habitasse platea dictumst. Morbi finibus tempor libero, id suscipit orci. Nulla feugiat felis vel est interdum, nec sodales eros volutpat. Aenean quis orci sapien. Quisque sed nulla finibus, fermentum erat vitae, posuere lacus. Pellentesque et sem sed sapien placerat eleifend sit amet at mauris. Aenean ornare augue ante, vel aliquet est suscipit non. Curabitur vel est et nisi imperdiet hendrerit nec in sem. In consequat blandit eros ac tincidunt. Nulla facilisi.

Duis vel neque at augue dignissim rhoncus. Donec arcu justo, accumsan porta interdum et, porttitor ut tortor. Integer lectus tortor, elementum ut quam a, dictum pretium metus. Morbi maximus fringilla massa, ac congue leo efficitur ut. Duis justo eros, facilisis ut eros porta, auctor consectetur sem. In pharetra massa vitae ornare ultricies. In pretium sapien ex, at aliquam turpis pellentesque in. Vestibulum sed risus sit amet mauris rutrum faucibus ut ut elit. Praesent in molestie metus, ac imperdiet sem.

Donec placerat tellus at odio dignissim scelerisque. Nulla pulvinar ipsum at est rutrum ullamcorper. Quisque nisl nibh, finibus fermentum quam non, dapibus bibendum metus. Phasellus blandit lacus a est suscipit, at mollis felis scelerisque. Vivamus eros orci, tincidunt non convallis ut, imperdiet non ipsum. Pellentesque non ornare leo, vulputate semper urna. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Donec vitae lobortis lorem, eget efficitur arcu. Vivamus suscipit orci enim, a vestibulum mi hendrerit at. Mauris sed ex eros.

Quisque tristique nibh at velit porttitor, vel varius sem mollis. Suspendisse at urna elit. Vivamus est elit, tincidunt a convallis et, blandit non orci. Pellentesque ac arcu et ex blandit viverra. Pellentesque id sodales libero, vel imperdiet est. Suspendisse potenti. Sed odio ligula, placerat id diam sit amet, luctus gravida nibh. Donec sed tempor lectus, vitae feugiat ex.

Morbi vitae arcu vel metus venenatis porta vel at massa. Sed efficitur nulla vitae euismod pretium. Morbi gravida diam sit amet feugiat imperdiet. Curabitur eros justo, ornare maximus turpis nec, vehicula ultricies quam. Phasellus rutrum quam eu libero egestas, sit amet cursus dui scelerisque. Phasellus non erat ut tortor pellentesque finibus non et diam. Sed nec vestibulum velit, et interdum ante. Quisque mollis, augue id egestas vehicula, libero orci varius ligula, vel elementum ante sapien in neque. Quisque ut commodo neque. Proin magna lorem, egestas vitae lacus et, pharetra ultricies lectus. Aliquam fermentum malesuada auctor.

Duis in tristique massa. Mauris consectetur nisi odio, non efficitur felis condimentum eget. Fusce porta risus sit amet sagittis blandit. Integer dictum ante diam, sed dapibus augue porta ac. Duis imperdiet scelerisque nunc in maximus. Proin feugiat condimentum est, id tristique nibh aliquet a. Aenean facilisis mattis quam, vitae porttitor dui tristique nec. Donec blandit sapien ac ex placerat, nec bibendum erat tincidunt. Pellentesque sollicitudin orci a libero aliquet, pretium aliquam nibh finibus. In dictum pulvinar sapien.

Aliquam elementum posuere magna, at placerat lectus molestie vitae. Nullam id convallis augue, vel sodales sapien. Fusce eu tincidunt orci, vel rutrum diam. Curabitur in lobortis mi, at fermentum lectus. Donec non accumsan est. Suspendisse mattis purus et nulla sodales luctus. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Cras eu rutrum lorem.

Sed aliquam, erat et tincidunt pulvinar, eros odio facilisis nisi, nec sollicitudin enim nisl ut urna. Sed aliquam ipsum placerat diam finibus, eget elementum quam faucibus. Interdum et malesuada fames ac ante ipsum primis in faucibus. Praesent vulputate nisl id justo facilisis gravida quis lobortis metus. Curabitur convallis efficitur ex, non bibendum diam maximus ac. Aliquam erat volutpat. Vivamus porttitor, sem id consequat dictum, arcu eros eleifend mi, ac maximus ipsum magna quis lacus. Suspendisse porta mollis diam, nec mollis massa commodo ac. Pellentesque non felis eget orci viverra blandit at in diam. In vestibulum laoreet enim vel ultrices. Sed gravida vestibulum nibh eu efficitur. Morbi vel pulvinar ante. Pellentesque non purus dapibus, condimentum eros eu, accumsan ante. Integer rhoncus rhoncus enim ut ornare. Mauris sagittis scelerisque turpis vestibulum imperdiet.

Phasellus commodo interdum enim. Fusce fringilla nisl id ipsum fringilla, in bibendum ex consectetur. Ut feugiat sem vitae elit rutrum fringilla. Vestibulum magna diam, lobortis a nulla at, egestas sagittis velit. Duis posuere nisl in efficitur suscipit. Vivamus in condimentum orci. Donec ut odio nisi. Cras dignissim et lacus sit amet mattis. Aliquam ut rutrum ligula. Nulla a nisi feugiat, aliquet risus vel, vulputate nunc. Nullam mollis massa vitae tortor fringilla malesuada. Phasellus eleifend at ante ac mattis. Donec eget tristique lacus, sed gravida quam. Sed ante odio, tempus vel nulla ac, fringilla gravida lacus. Proin pellentesque neque eget lorem porttitor, eget dictum purus semper.

Quisque in feugiat mi, ut pellentesque nulla. Sed convallis aliquet ligula eu sagittis. Etiam in semper sapien. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Curabitur eleifend pretium lectus in faucibus. Vestibulum quam dui, lobortis eget odio vitae, ultrices tempor felis. Vivamus finibus consectetur ante, ut posuere lectus malesuada at. Praesent consequat est quis erat interdum accumsan. Praesent purus libero, euismod a nulla vitae, fringilla hendrerit libero. Quisque eu feugiat enim. Maecenas eget mi mauris. Donec et sollicitudin quam, vitae tristique orci. Nullam sit amet diam sit amet arcu volutpat venenatis.

Aenean molestie non tortor at gravida. Phasellus a est vitae enim auctor tristique. Donec et nunc dolor. Pellentesque nibh augue, convallis eu rutrum vitae, lobortis sit amet sem. Maecenas nec egestas lacus. Phasellus erat est, pretium sed tellus id, porta accumsan ipsum. Phasellus justo leo, maximus sit amet nibh eget, mattis semper libero. Vestibulum elementum mauris quis risus bibendum, non pulvinar justo vulputate. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Fusce finibus ultrices enim, vitae scelerisque urna cursus a. Maecenas vulputate mi a ligula mollis dictum. Pellentesque eget euismod dolor. Aliquam tempor nulla quam, eu vulputate lorem faucibus vel. Nunc elementum libero consequat hendrerit imperdiet.

Vivamus ac porta purus. Donec ac nisl nec nulla volutpat sagittis a vel elit. Donec vitae dapibus neque. Morbi dolor enim, condimentum at magna quis, aliquet interdum est. Vestibulum ut velit at velit facilisis varius vel eu mauris. Etiam cursus fermentum nunc, vitae fermentum odio ultricies eu. Vivamus sit amet commodo orci, id ullamcorper lacus. Integer varius, tellus sed tristique ultrices, nisi sapien auctor quam, sit amet feugiat lectus nibh eget elit. Etiam quis porttitor odio, at interdum augue. Donec eu mauris at augue auctor cursus. Sed justo ipsum, pulvinar a massa ut, accumsan molestie nunc. Interdum et malesuada fames ac ante ipsum primis in faucibus.

Morbi tristique eu erat quis vulputate. Sed quam ante, vehicula in libero sed, accumsan tristique elit. Phasellus egestas nibh arcu, vehicula rutrum nunc tristique nec. Fusce ac iaculis erat. Curabitur quis condimentum sem, ac egestas dui. Mauris dictum lorem eget molestie laoreet. Quisque eu neque lectus. Maecenas a gravida dolor. In scelerisque est sit amet vestibulum efficitur. Proin non sollicitudin diam. Quisque vulputate leo ut metus consequat sagittis. Donec gravida purus vitae felis sollicitudin aliquet. Sed finibus justo convallis, tempor arcu vitae, tincidunt metus. Ut scelerisque lorem sit amet pharetra faucibus. """

def create_model(model_dir):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        load_in_4bit=True,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return pipe


def generate(pipe, text, **kwargs):
    prompt = f"<s>[INST]translate the following text into english: {text}[/INST]"
    results = pipe(
        text,
        return_full_text=False,
        **kwargs,
    )
    return results[0]["generated_text"]


# def generate(text, max_tokens: int=100, temperature=0.7, top_k=None, top_p=None):
#     sampling_args = {max_tokens: max_tokens, temperature: temperature}
#     if top_k is not None:
#         sampling_args["top_k"] = top_k
#     if top_p is not None:
#         sampling_args["top_p"] = top_p
#     sampling_params = SamplingParams(**sampling_args)
#     results = model.generate(text, sampling_params=sampling_params)
#     breakpoint()
#     for result in results:
#         yield result.outputs[0].text


def interact(pipe):
    max_tokens = 100
    temperature = 0.7
    while True:
        text = input("Enter text: ")
        if text.startswith("/"):
            match text[1:]:
                case "exit":
                    return
                case "quit":
                    return
                case "q":
                    return
                case "stop":
                    return
                case "temperature":
                    try:
                        temperature = float(text.split(" ")[1])
                    except ValueError:
                        print("Error: Invalid value passed to temperature.")
                        continue
                case "max_tokens":
                    try:
                        max_tokens = int(text.split(" ")[1])
                    except ValueError:
                        print("Error: Invalid value passed to max_tokens.")
                        continue
                case "break":
                    breakpoint()
                case _:
                    print("Unknown command")
            continue
        result = generate(
            pipe,
            text,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            do_sample=True,
        ).replace("\\n", "\n")
        print(result)

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time() - start
        return result, end
    return wrapper

def test_profile(pipe):
    timed_generate =  timeit(functools.partial(generate, pipe))
    length = 3000
    max_tokens = 10
    text = lorem_ipsum[:100]
    timed_generate(text, max_new_tokens=max_tokens)
    lorem_ids = pipe.tokenizer.encode(lorem_ipsum)
    input_ids = lorem_ids[:length]
    text = pipe.tokenizer.decode(input_ids, skip_special_tokens=True)
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            output, runtime = timed_generate(text, max_new_tokens=max_tokens, min_new_tokens=max_tokens)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")
    result = (len(input_ids), len(pipe.tokenizer.encode(output)), runtime)
    print(result)



def test_input_variation(pipe):
    timed_generate =  timeit(functools.partial(generate, pipe))
    max_tokens = 1
    text = lorem_ipsum[:100]
    timed_generate(text, max_new_tokens=max_tokens)
    timed_generate(text, max_new_tokens=max_tokens)
    lorem_ids = pipe.tokenizer.encode(lorem_ipsum)
    texts = []
    results = []
    try:
        for length in range(100, 3000, 300):
            input_ids = lorem_ids[:length]
            text = pipe.tokenizer.decode(input_ids, skip_special_tokens=True)
            texts.append(text)
            output, runtime = timed_generate(text, max_new_tokens=max_tokens, min_new_tokens=max_tokens)
            # print(input_ids, text, output)
            result = (len(input_ids), len(pipe.tokenizer.encode(output)), runtime)
            results.append(result)
            print(result)
    finally:
        save_test_results(results, "input_variation.csv")
    test_profile(pipe)

def test_output_variation(pipe):
    timed_generate =  timeit(functools.partial(generate, pipe))
    max_tokens = 100
    min_tokens = max_tokens
    text = lorem_ipsum[:100]
    timed_generate(text, max_new_tokens=max_tokens)
    timed_generate(text, max_new_tokens=max_tokens)
    lorem_ids = pipe.tokenizer.encode(lorem_ipsum)
    input_ids = lorem_ids[:100]
    text = pipe.tokenizer.decode(input_ids, skip_special_tokens=True)
    results = []
    for length in range(100, 4000, 299):
        max_tokens = length
        min_tokens = length
        output, runtime = timed_generate(text, max_new_tokens=max_tokens, min_new_tokens=min_tokens)
        # print(input_ids, text, output)
        result = (len(input_ids), len(pipe.tokenizer.encode(output)), runtime)
        print(result)
        results.append(result)
    save_test_results(results, "output_variation.csv")

def save_test_results(results, filename):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)

def run_test(model_dir):
    pipe = create_model(model_dir)
    # test_input_variation(pipe)
    test_profile(pipe)

if __name__ == "__main__":
    fire.Fire(run_test)
