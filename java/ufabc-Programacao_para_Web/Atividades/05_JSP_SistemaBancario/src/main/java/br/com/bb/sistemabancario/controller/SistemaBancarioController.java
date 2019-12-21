package br.com.bb.sistemabancario.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.servlet.ModelAndView;

import org.springframework.web.servlet.mvc.support.RedirectAttributes;
import org.springframework.web.servlet.view.RedirectView;
import br.com.bb.sistemabancario.model.entity.ContaCorrente;
import br.com.bb.sistemabancario.service.ContaCorrenteService;


/*Controller - Intermediação entre Model e View - Porta de entrada do sistema.
 * 
 * Gerenciamento de interações com usuários através da View e dispara atualizações para a Model, resolvendo
 * as requisições e devolvendo um resultado ao requisitor.
 * 
 * Servlets que respondem às requisições HTTP selecionando a View de resposta para determinada ação.
 * 
 */

@Controller
public class SistemaBancarioController {

	@Autowired
	private ContaCorrenteService ccService;
	
	//- - - CONTROLLER FOR JSP - - -
	@RequestMapping(value = { "/", "/index" })
	@ResponseBody
	public ModelAndView index() {
		ModelAndView mv = new ModelAndView("index");
		mv.addObject("ccs", ccService.getAll());
		return mv;
	}
	
	@RequestMapping("/insere")
	public String salvar()	{	
		return "insere";
	}
	
	@RequestMapping(value="/save", method=RequestMethod.POST)
	public RedirectView insere(ContaCorrente cc, Model model, RedirectAttributes attributes) {
		ccService.create(cc);
		attributes.addFlashAttribute("msg", "Conta corrente: " + cc.getNumero()
		+ " inserida com sucesso!");
		return new RedirectView("/");
	}	
	
	//- - - CONTROLLER FOR HTML - - -
	@RequestMapping("/teste")
	public ModelAndView teste() {
		return new ModelAndView("teste");
	}
	
	@GetMapping("/inserir")
	public ModelAndView recebeInsere(Model model) {
		model.addAttribute("contacorrente", new ContaCorrente());
		return new ModelAndView("inserircc");		
	}
	
	@PostMapping("/inserir")
	public ModelAndView salvaInsere(@ModelAttribute("contacorrente") ContaCorrente cc, Model model) {
		ccService.create(cc);
		return new ModelAndView("inserircc");
	}
	 
	/*@Autowired
	private ComentarioService comentService;
	@Autowired
	private NoticiaService noticiaService;
	@Autowired
	private TipoNoticiaService tipoService;
	
	@RequestMapping("/")
	public ModelAndView index() {
		return new ModelAndView("index");
	}
	
	@RequestMapping("/listarcoment")
	public ModelAndView listar(Model model) {
		model.addAttribute("comentario", comentService.getAll());
		return new ModelAndView("listarcomentarios");
	}
	
	@GetMapping("/adicionarcoment")
	public ModelAndView recebeInserir(Model model) {
		model.addAttribute("comentario", new Comentario());
		return new ModelAndView("inserircomentario");
	}
	
	@PostMapping("/adicionarcoment")
	public ModelAndView salvaInserir(@ModelAttribute("comentario") Comentario coment, Model model) {
		comentService.create(coment);
		return new ModelAndView("inserircomentario");
	}*/
}

